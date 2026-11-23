"""Neural multitask query analyzer: retrieval necessity + temporal class + confidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from .hf_utils import from_pretrained_local_first, prefer_local_hf_cache
from .model_config import DEFAULT_ANALYZER_MODEL, LEGACY_ANALYZER_MODEL

TEMPORAL_LABELS = ["current", "historical", "timeless"]
LABEL2ID = {l: i for i, l in enumerate(TEMPORAL_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def _encoder_hidden_size(config: Any) -> int:
    return int(getattr(config, "hidden_size", getattr(config, "dim", 768)))


def _select_training_precision(use_cuda: bool) -> tuple[str, Optional[torch.dtype]]:
    """T4/V100: full fp32 (stable for DeBERTa). A100/L4+: optional bf16 autocast."""
    if not use_cuda:
        return "fp32", None
    gpu = torch.cuda.get_device_name(0).lower()
    if any(tag in gpu for tag in ("a100", "h100", "l40", "l4", "h200")):
        if torch.cuda.is_bf16_supported():
            return "bf16-autocast", torch.bfloat16
    return "fp32", None


def _freeze_encoder_layers(encoder: nn.Module, freeze_layers: int) -> None:
    if freeze_layers <= 0:
        return
    layers = None
    tr = getattr(encoder, "transformer", None)
    if tr is not None and hasattr(tr, "layer"):
        layers = tr.layer
    enc = getattr(encoder, "encoder", None)
    if layers is None and enc is not None and hasattr(enc, "layer"):
        layers = enc.layer
    if layers is None:
        print("Warning: freeze_layers ignored — unknown encoder layout")
        return
    for layer in layers[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad = False


def _inverse_freq_class_weights(counts: dict[int, int], n_classes: int) -> torch.Tensor:
    """Balanced CE weights: total / (n_classes * count_c)."""
    total = sum(counts.get(i, 0) for i in range(n_classes)) or 1
    weights = []
    for i in range(n_classes):
        c = max(counts.get(i, 0), 1)
        weights.append(total / (n_classes * c))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * n_classes

_AUGMENT_PREFIXES = [
    "",
    "Please answer: ",
    "Quickly: ",
    "Tell me: ",
    "I need to know: ",
    "Question — ",
    "As asked: ",
]


def augment_training_rows(
    texts: list[str],
    needs: list[int],
    temporal: list[str],
    factor: int = 1,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str]]:
    """Expand train rows with light paraphrase prefixes (labels unchanged).

    factor=1 keeps data as-is. factor=4 → roughly 4x train examples for longer GPU runs.
    """
    factor = max(1, int(factor))
    if factor == 1:
        return texts, needs, temporal
    import random

    rng = random.Random(seed)
    out_t: list[str] = []
    out_n: list[int] = []
    out_c: list[str] = []
    for text, n, c in zip(texts, needs, temporal):
        for i in range(factor):
            if i == 0:
                out_t.append(text)
            else:
                prefix = rng.choice(_AUGMENT_PREFIXES[1:])
                suffix = rng.choice(["", "?", " Please be brief."])
                base = text.rstrip("?")
                out_t.append(f"{prefix}{base}{suffix}".strip())
            out_n.append(n)
            out_c.append(c)
    return out_t, out_n, out_c


@dataclass
class AnalyzerOutput:
    needs_retrieval: int
    temporal_class: str
    confidence: float
    necessity_prob: float
    temporal_probs: dict[str, float]


class QueryDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        needs: list[int],
        temporal_ids: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ) -> None:
        self.texts = texts
        self.needs = needs
        self.temporal_ids = temporal_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "needs": torch.tensor(self.needs[idx], dtype=torch.long),
            "temporal": torch.tensor(self.temporal_ids[idx], dtype=torch.long),
        }


class MultitaskQueryAnalyzer(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_ANALYZER_MODEL,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        temporal_loss_weight: float = 1.75,
    ) -> None:
        super().__init__()
        prefer_local_hf_cache()
        self.model_name = model_name
        self.label_smoothing = label_smoothing
        self.temporal_loss_weight = temporal_loss_weight
        enc_kwargs: dict[str, Any] = {}
        if "deberta" in model_name.lower():
            enc_kwargs["attn_implementation"] = "eager"
        self.encoder = from_pretrained_local_first(AutoModel.from_pretrained, model_name, **enc_kwargs)
        hidden = _encoder_hidden_size(self.encoder.config)
        self.dropout = nn.Dropout(dropout)
        self.necessity_head = nn.Linear(hidden, 2)
        self.temporal_head = nn.Linear(hidden, len(TEMPORAL_LABELS))
        self.confidence_head = nn.Linear(hidden, 1)
        for head in (self.necessity_head, self.temporal_head, self.confidence_head):
            nn.init.xavier_uniform_(head.weight, gain=0.02)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        needs: Optional[torch.Tensor] = None,
        temporal: Optional[torch.Tensor] = None,
        nec_class_weight: Optional[torch.Tensor] = None,
        temp_class_weight: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Heads stay fp32; avoids bf16/fp32 mismatch in Linear on T4 + autocast.
        cls = self.dropout(out.last_hidden_state[:, 0]).float()
        nec_logits = self.necessity_head(cls)
        temp_logits = self.temporal_head(cls)
        conf_logits = self.confidence_head(cls).squeeze(-1)
        conf = torch.sigmoid(conf_logits)

        loss = None
        if needs is not None and temporal is not None:
            ls = self.label_smoothing
            nec_loss = nn.functional.cross_entropy(
                nec_logits, needs, weight=nec_class_weight, label_smoothing=ls
            )
            temp_loss = nn.functional.cross_entropy(
                temp_logits, temporal, weight=temp_class_weight, label_smoothing=ls
            )
            loss = nec_loss + self.temporal_loss_weight * temp_loss

        return {
            "loss": loss,
            "necessity_logits": nec_logits,
            "temporal_logits": temp_logits,
            "confidence": conf,
        }


class QueryAnalyzerService:
    def __init__(
        self,
        model: MultitaskQueryAnalyzer,
        tokenizer: PreTrainedTokenizerBase,
        device: Optional[str] = None,
        max_length: int = 64,
        conf_threshold: float = 0.55,
        temperature: float = 1.0,
        blend_confidence: bool = True,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conf_threshold = conf_threshold
        self.temperature = temperature
        self.blend_confidence = blend_confidence

    @classmethod
    def from_pretrained_bundle(
        cls,
        artifacts_dir: str | Path,
        device: Optional[str] = None,
    ) -> "QueryAnalyzerService":
        artifacts_dir = Path(artifacts_dir)
        weights = artifacts_dir / "analyzer.pt"
        if not weights.exists():
            raise FileNotFoundError(
                f"Missing trained analyzer at {weights}. Run train_analyzer / Colab section 3 first."
            )
        prefer_local_hf_cache()
        meta = json.loads((artifacts_dir / "analyzer_meta.json").read_text(encoding="utf-8"))
        model_name = meta.get("model_name", LEGACY_ANALYZER_MODEL)
        tok_name = meta.get("tokenizer_name", model_name)
        tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, tok_name)
        model = MultitaskQueryAnalyzer(model_name=model_name)
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=meta.get("max_length", 64),
            conf_threshold=meta.get("conf_threshold", 0.55),
        )

    def predict(self, query: str) -> AnalyzerOutput:
        enc = self.tokenizer(
            query,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        temperature = max(float(self.temperature), 1e-3)
        nec_probs = torch.softmax(out["necessity_logits"] / temperature, dim=-1)[0]
        temp_probs = torch.softmax(out["temporal_logits"] / temperature, dim=-1)[0]
        needs = int(nec_probs.argmax().item())
        temp_id = int(temp_probs.argmax().item())
        conf = float(out["confidence"][0].item())
        max_prob = float(max(nec_probs.max().item(), temp_probs.max().item()))
        if self.blend_confidence:
            confidence = 0.5 * conf + 0.5 * max_prob
        else:
            confidence = max_prob
        return AnalyzerOutput(
            needs_retrieval=needs,
            temporal_class=ID2LABEL[temp_id],
            confidence=confidence,
            necessity_prob=float(nec_probs[1].item()),
            temporal_probs={ID2LABEL[i]: float(temp_probs[i].item()) for i in range(len(TEMPORAL_LABELS))},
        )

    def save(self, artifacts_dir: str | Path, extra_meta: Optional[dict[str, Any]] = None) -> None:
        artifacts_dir = Path(artifacts_dir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), artifacts_dir / "analyzer.pt")
        meta = {
            "model_name": self.model.model_name,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", self.model.model_name),
            "max_length": self.max_length,
            "conf_threshold": self.conf_threshold,
            "temporal_labels": TEMPORAL_LABELS,
        }
        if extra_meta:
            meta.update(extra_meta)
        (artifacts_dir / "analyzer_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _training_preflight(
    model: MultitaskQueryAnalyzer,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    max_length: int,
) -> None:
    """Fail fast if forward/loss is non-finite before a long Colab run."""
    model.train()
    enc = tokenizer(
        "Who is the current Prime Minister of India?",
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    needs = torch.tensor([1], device=device)
    temporal = torch.tensor([0], device=device)
    out = model(
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        needs=needs,
        temporal=temporal,
    )
    loss = out["loss"]
    if loss is None or not torch.isfinite(loss).item():
        raise RuntimeError(
            "Training preflight failed (non-finite loss). "
            "On T4 use distilbert-base-uncased; on A100 try DeBERTa with LR=1e-5."
        )
    print(f"Preflight OK — sample loss={float(loss.item()):.4f}")


def train_analyzer(
    train_texts: list[str],
    train_needs: list[int],
    train_temporal: list[str],
    val_texts: list[str],
    val_needs: list[int],
    val_temporal: list[str],
    artifacts_dir: str | Path,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float | None = None,
    max_length: int = 64,
    model_name: str = DEFAULT_ANALYZER_MODEL,
    freeze_layers: int = 2,
    device: Optional[str] = None,
    augment_factor: int = 1,
    patience: int = 5,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.0,
    gradient_checkpointing: bool = False,
    use_weighted_sampler: bool = False,
    init_state: str | Path | None = None,
) -> dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if lr is None:
        lr = 1e-5 if "deberta" in model_name.lower() else 2e-5
    prefer_local_hf_cache()
    tokenizer = from_pretrained_local_first(AutoTokenizer.from_pretrained, model_name)
    model = MultitaskQueryAnalyzer(
        model_name=model_name,
        label_smoothing=label_smoothing,
    ).to(device)
    if init_state and Path(init_state).exists():
        state = torch.load(init_state, map_location="cpu")
        model.load_state_dict(state)
        print(f"Initialized analyzer from {init_state}")
    if gradient_checkpointing and hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    _freeze_encoder_layers(model.encoder, freeze_layers)
    model.float()

    nec_counts: dict[int, int] = {}
    temp_counts: dict[int, int] = {}
    for n, t in zip(train_needs, train_temporal):
        nec_counts[n] = nec_counts.get(n, 0) + 1
        tid = LABEL2ID[t]
        temp_counts[tid] = temp_counts.get(tid, 0) + 1
    nec_w = _inverse_freq_class_weights(nec_counts, 2).to(device)
    temp_w = _inverse_freq_class_weights(temp_counts, len(TEMPORAL_LABELS)).to(device)
    print(f"Train label balance — needs: {nec_counts} temporal: {temp_counts}")

    train_texts, train_needs, train_temporal = augment_training_rows(
        train_texts, train_needs, train_temporal, factor=augment_factor
    )
    print(f"Train examples after augment_factor={augment_factor}: {len(train_texts)}")

    train_temporal_ids = [LABEL2ID[t] for t in train_temporal]
    train_ds = QueryDataset(
        train_texts,
        train_needs,
        train_temporal_ids,
        tokenizer,
        max_length=max_length,
    )
    val_ds = QueryDataset(
        val_texts,
        val_needs,
        [LABEL2ID[t] for t in val_temporal],
        tokenizer,
        max_length=max_length,
    )
    use_cuda = device.startswith("cuda")
    # num_workers>0 often hangs in Colab notebooks; keep 0 for reliable training.
    n_workers = 0
    if use_weighted_sampler:
        sample_weights = [1.0 / max(temp_counts[tid], 1) for tid in train_temporal_ids]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=False,
            pin_memory=use_cuda,
            num_workers=n_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_cuda,
            num_workers=n_workers,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        pin_memory=use_cuda,
        num_workers=n_workers,
    )

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        eps=1e-6,
    )
    total_steps = max(len(train_loader) * epochs, 1)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * total_steps), total_steps)
    amp_mode, amp_dtype = _select_training_precision(use_cuda)

    history: list[dict[str, float]] = []
    best_val = -1.0
    best_state = None
    steps_per_epoch = len(train_loader)
    print(
        f"Training {model_name} on {device} · batch={batch_size} · "
        f"precision={amp_mode} · freeze={freeze_layers} · patience={patience}"
    )
    print(
        f"Steps/epoch={steps_per_epoch} (total~{steps_per_epoch * epochs}) — "
        "first epoch line may take 10–25 min on T4; GPU util should be >0%."
    )

    _training_preflight(model, tokenizer, device, max_length)

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    epochs_without_improve = 0
    nan_skips = 0
    max_nan_skips = 50
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_loss_batches = 0
        batch_iter = train_loader
        if _tqdm is not None:
            batch_iter = _tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in batch_iter:
            optim.zero_grad(set_to_none=True)
            autocast_enabled = amp_dtype is not None
            with torch.amp.autocast(
                "cuda",
                enabled=autocast_enabled,
                dtype=amp_dtype or torch.float32,
            ):
                out = model(
                    input_ids=batch["input_ids"].to(device, non_blocking=use_cuda),
                    attention_mask=batch["attention_mask"].to(device, non_blocking=use_cuda),
                    needs=batch["needs"].to(device, non_blocking=use_cuda),
                    temporal=batch["temporal"].to(device, non_blocking=use_cuda),
                    nec_class_weight=nec_w,
                    temp_class_weight=temp_w,
                )
                loss = out["loss"]
            loss_val = float(loss.detach().item())
            if not torch.isfinite(loss).item():
                nan_skips += 1
                if nan_skips <= 3:
                    print(f"WARN: non-finite loss at epoch {epoch} (skip #{nan_skips}) — skipping step.")
                if nan_skips >= max_nan_skips:
                    raise RuntimeError(
                        f"Training aborted: {nan_skips} non-finite losses. "
                        "Try LR=5e-6, BATCH_SIZE=4, FREEZE_LAYERS=4, or distilbert-base-uncased."
                    )
                optim.zero_grad(set_to_none=True)
                continue
            try:
                loss.backward()
            except RuntimeError as exc:
                if use_cuda and "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "CUDA OOM during training — lower BATCH_SIZE (e.g. 4) or MAX_LENGTH (e.g. 96)."
                    ) from exc
                raise
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            total_loss += loss_val
            n_loss_batches += 1

        metrics = evaluate_model(model, val_loader, device)
        metrics["train_loss"] = total_loss / max(n_loss_batches, 1)
        metrics["epoch"] = epoch
        history.append(metrics)
        score = 0.25 * metrics["necessity_f1"] + 0.75 * metrics["temporal_macro_f1"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        print(
            f"Epoch {epoch}: loss={metrics['train_loss']:.4f} "
            f"val_nec_f1={metrics['necessity_f1']:.3f} val_nec_acc={metrics['necessity_acc']:.3f} "
            f"val_temp_acc={metrics['temporal_acc']:.3f} val_temp_f1={metrics['temporal_macro_f1']:.3f}"
        )
        if patience > 0 and epochs_without_improve >= patience:
            print(f"Early stopping — no val improvement for {patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    service = QueryAnalyzerService(model=model, tokenizer=tokenizer, device=device, max_length=max_length)
    service.save(
        artifacts_dir,
        extra_meta={
            "history": history,
            "best_val_score": best_val,
            "augment_factor": augment_factor,
            "train_examples": len(train_texts),
            "epochs": epochs,
            "batch_size": batch_size,
            "model_name": model_name,
        },
    )
    return {"history": history, "best_val_score": best_val, "train_examples": len(train_texts)}


def load_query_analyzer(
    artifacts_dir: str | Path,
    device: Optional[str] = None,
) -> QueryAnalyzerService:
    """Load the trained neural query analyzer (required for Adaptive + fair eval)."""
    return QueryAnalyzerService.from_pretrained_bundle(artifacts_dir, device=device)


def load_router_v2(artifacts_dir: str | Path, device: Optional[str] = None) -> QueryAnalyzerService:
    """Load the calibrated router. Does not use the untrained confidence head."""
    artifacts_dir = Path(artifacts_dir)
    weights = artifacts_dir / "analyzer_v2.pt"
    meta_path = artifacts_dir / "analyzer_v2_meta.json"
    if not weights.exists():
        raise FileNotFoundError(f"Missing {weights}. Run scripts/train_router_v2.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_name = meta.get("model_name", DEFAULT_ANALYZER_MODEL)
    tokenizer = from_pretrained_local_first(
        AutoTokenizer.from_pretrained, meta.get("tokenizer_name", model_name)
    )
    model = MultitaskQueryAnalyzer(model_name=model_name)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    service = QueryAnalyzerService(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=meta.get("max_length", 64),
        conf_threshold=meta.get("conf_threshold", 0.55),
        temperature=float(meta.get("temperature", 1.0)),
        blend_confidence=False,
    )
    return service


def evaluate_model(model: MultitaskQueryAnalyzer, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    nec_tp = nec_fp = nec_fn = 0
    nec_correct = 0
    temp_correct = 0
    n = 0
    n_classes = len(TEMPORAL_LABELS)
    temp_tp = [0] * n_classes
    temp_fp = [0] * n_classes
    temp_fn = [0] * n_classes
    with torch.no_grad():
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            nec_pred = out["necessity_logits"].argmax(-1).cpu()
            temp_pred = out["temporal_logits"].argmax(-1).cpu()
            needs = batch["needs"]
            temporal = batch["temporal"]
            for p, y in zip(nec_pred.tolist(), needs.tolist()):
                if p == y:
                    nec_correct += 1
                if p == 1 and y == 1:
                    nec_tp += 1
                elif p == 1 and y == 0:
                    nec_fp += 1
                elif p == 0 and y == 1:
                    nec_fn += 1
            for p, y in zip(temp_pred.tolist(), temporal.tolist()):
                if p == y:
                    temp_correct += 1
                for c in range(n_classes):
                    if p == c and y == c:
                        temp_tp[c] += 1
                    elif p == c and y != c:
                        temp_fp[c] += 1
                    elif p != c and y == c:
                        temp_fn[c] += 1
            n += len(needs)

    prec = nec_tp / (nec_tp + nec_fp + 1e-8)
    rec = nec_tp / (nec_tp + nec_fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1s = []
    for c in range(n_classes):
        p = temp_tp[c] / (temp_tp[c] + temp_fp[c] + 1e-8)
        r = temp_tp[c] / (temp_tp[c] + temp_fn[c] + 1e-8)
        f1s.append(2 * p * r / (p + r + 1e-8))
    return {
        "necessity_f1": float(f1),
        "necessity_acc": float(nec_correct / max(n, 1)),
        "necessity_precision": float(prec),
        "necessity_recall": float(rec),
        "temporal_acc": float(temp_correct / max(n, 1)),
        "temporal_macro_f1": float(sum(f1s) / max(len(f1s), 1)),
    }



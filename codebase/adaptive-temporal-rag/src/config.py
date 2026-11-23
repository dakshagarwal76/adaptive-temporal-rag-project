"""Frozen evaluation config (Checkpoint B)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "final.yaml"


@dataclass
class EvalConfig:
    name: str = "checkpoint_b_final"
    generator: str = "extractive"
    top_k: int = 5
    candidate_k: int = 80
    ce_rerank_keep: int = 40
    alpha: float = 0.6
    conf_threshold: float = 0.55
    sufficiency_threshold: float = 0.55
    max_sufficiency_rounds: int = 2
    query_timeout_s: float = 60.0
    enable_ce_rerank: bool = True
    enable_sufficiency_loop: bool = True
    enable_adaptive_window: bool = True
    enable_entity_cut: bool = True
    enable_sentence_selector: bool = True
    enable_answer_projection: bool = True
    always_retrieve: bool = False
    span_extract_timeqa: bool = True
    learned_experimental: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_ablation(self, name: str) -> "EvalConfig":
        cfg = EvalConfig(**self.as_dict())
        if name == "router_off":
            cfg.always_retrieve = True
        elif name == "window_off":
            cfg.enable_adaptive_window = False
        elif name == "sufficiency_off":
            cfg.enable_sufficiency_loop = False
        elif name == "entity_cut_off":
            cfg.enable_entity_cut = False
        elif name == "sentence_selector_off":
            cfg.enable_sentence_selector = False
        elif name:
            raise ValueError(f"Unknown ablation {name}")
        return cfg


def _parse_scalar(raw: str) -> Any:
    text = raw.split("#", 1)[0].strip().strip("'\"")
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_eval_config(path: str | Path | None = None) -> EvalConfig:
    target = Path(path) if path else DEFAULT_CONFIG
    data: dict[str, Any] = {}
    if target.exists():
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            for line in target.read_text(encoding="utf-8").splitlines():
                stripped = line.split("#", 1)[0].strip()
                if not stripped or ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                data[key.strip()] = _parse_scalar(value)
    allowed = {field.name for field in EvalConfig.__dataclass_fields__.values()}
    return EvalConfig(**{k: v for k, v in data.items() if k in allowed})

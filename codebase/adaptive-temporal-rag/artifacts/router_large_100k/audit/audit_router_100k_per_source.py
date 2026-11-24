"""Per-source + labeler-rule + counterfactual audit of the 100k DeBERTa-v3-base router.
Uses the repo's own QueryAnalyzerService.predict math (temperature from meta, blend_confidence)."""
import json, sys, time, collections, hashlib, re, platform
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src import router_large as RL
from src.router_large import load_large_router, canonicalize
from sklearn.metrics import f1_score, confusion_matrix

LABELS = ["current", "historical", "timeless"]
torch.manual_seed(0); torch.set_num_threads(8)
BUNDLE = Path("/workspace/kaggle_router/extracted_v1")
svc = load_large_router(BUNDLE, device="cpu")
print("loaded", svc.temperature, svc.blend_confidence, svc.max_length, flush=True)

def predict_batch(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        enc = svc.tokenizer(texts[i:i+bs], truncation=True, padding="max_length", max_length=svc.max_length, return_tensors="pt")
        with torch.no_grad():
            o = svc.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        T = max(float(svc.temperature), 1e-3)
        nec = torch.softmax(o["necessity_logits"] / T, -1)
        tmp = torch.softmax(o["temporal_logits"] / T, -1)
        conf = o["confidence"].view(-1)
        for j in range(len(nec)):
            mp = float(max(nec[j].max(), tmp[j].max()))
            c = 0.5 * float(conf[j]) + 0.5 * mp if svc.blend_confidence else mp
            out.append({"needs": int(nec[j].argmax()), "temporal": LABELS[int(tmp[j].argmax())],
                        "necessity_prob": float(nec[j][1]), "temporal_probs": [float(x) for x in tmp[j]], "confidence": c})
        if (i // bs) % 20 == 0: print(f"  {i+len(nec)}/{len(texts)}", flush=True)
    return out

def rule_fired(question, source):
    """Which branch of router_large.label_question produced the silver label (same order)."""
    text = " ".join((question or "").split()); can = canonicalize(text)
    years = [int(m.group(0)) for m in RL._YEAR_RE.finditer(can)]
    if years: return "explicit_year"
    if RL._CURRENT_RE.search(can): return "current_cue_regex"
    if RL._REL_PAST_RE.search(can): return "relative_past_regex"
    if RL._WHEN_RE.search(can): return "when_regex"
    if RL._OFFICE_NOW_RE.search(can) and not RL._OFFICE_PAST_RE.search(can): return "office_now_regex"
    if RL._OFFICE_PAST_RE.search(can): return "office_past_regex"
    return "source_default:" + (source or "").lower()

def metrics(gold_t, pred_t, gold_n, pred_n):
    present = sorted(set(gold_t) | set(pred_t), key=LABELS.index)
    return {"n": len(gold_t),
            "temporal_accuracy": float(np.mean([a == b for a, b in zip(gold_t, pred_t)])),
            "temporal_macro_f1_present_labels": float(f1_score(gold_t, pred_t, labels=present, average="macro", zero_division=0)),
            "necessity_accuracy": float(np.mean([a == b for a, b in zip(gold_n, pred_n)])),
            "n_temporal_errors": int(sum(a != b for a, b in zip(gold_t, pred_t))),
            "gold_label_counts": dict(collections.Counter(gold_t)),
            "temporal_confusion_rows_gold_cols_pred": {"labels": LABELS, "matrix": confusion_matrix(gold_t, pred_t, labels=LABELS).tolist()}}

rows = [json.loads(l) for l in open(ROOT / "data/test_100k.jsonl")]
t0 = time.time(); preds = predict_batch([r["query"] for r in rows]); infer_s = time.time() - t0
gt = [r["temporal_class"] for r in rows]; pt = [p["temporal"] for p in preds]
gn = [int(r["needs_retrieval"]) for r in rows]; pn = [p["needs"] for p in preds]
overall = metrics(gt, pt, gn, pn)
overall["temporal_macro_f1_all3"] = float(f1_score(gt, pt, labels=LABELS, average="macro"))
overall["necessity_macro_f1"] = float(f1_score(gn, pn, average="macro"))

# labeler reproduction check
relabel = [RL.label_question(r["query"], r["source"]) for r in rows]
relabel_match = sum(1 for r, l in zip(rows, relabel) if l and l["temporal_class"] == r["temporal_class"])
rules = [rule_fired(r["query"], r["source"]) for r in rows]

per_source = {}
for s in sorted({r["source"] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r["source"] == s]
    m = metrics([gt[i] for i in idx], [pt[i] for i in idx], [gn[i] for i in idx], [pn[i] for i in idx])
    cnt = collections.Counter(gt[i] for i in idx)
    m["majority_label"], m["majority_share_purity"] = cnt.most_common(1)[0][0], cnt.most_common(1)[0][1] / len(idx)
    m["single_class"] = len(cnt) == 1
    m["silver_label_rule_counts"] = dict(collections.Counter(rules[i] for i in idx))
    per_source[s] = m
per_rule = {}
for rname in sorted(set(rules)):
    idx = [i for i, x in enumerate(rules) if x == rname]
    per_rule[rname] = metrics([gt[i] for i in idx], [pt[i] for i in idx], [gn[i] for i in idx], [pn[i] for i in idx])
per_label_source = {lab: dict(collections.Counter(r["source"] for r in rows if r["temporal_class"] == lab)) for lab in LABELS}
src_major = {s: per_source[s]["majority_label"] for s in per_source}
source_only_acc = float(np.mean([src_major[r["source"]] == r["temporal_class"] for r in rows]))
src_only_f1 = float(f1_score(gt, [src_major[r["source"]] for r in rows], labels=LABELS, average="macro", zero_division=0))
regex_rule_rows = sum(1 for x in rules if not x.startswith("source_default"))
errors = [{"query": rows[i]["query"], "source": rows[i]["source"], "gold": gt[i], "pred": pt[i], "rule": rules[i],
           "temporal_probs": dict(zip(LABELS, [round(x, 4) for x in preds[i]["temporal_probs"]]))} for i in range(len(rows)) if gt[i] != pt[i]]

# counterfactual probe (hand-written, gold = semantic judgement of the evaluator, NOT the regex labeler)
probe = json.load(open(ROOT / "probe.json"))
pp = predict_batch([q["query"] for q in probe])
probe_rows = []
for q, p in zip(probe, pp):
    lab = RL.label_question(q["query"], "nq_open")
    probe_rows.append({**q, "pred_temporal": p["temporal"], "pred_needs": p["needs"],
                       "temporal_probs": dict(zip(LABELS, [round(x, 4) for x in p["temporal_probs"]])),
                       "confidence": round(p["confidence"], 4),
                       "silver_labeler_temporal": lab["temporal_class"] if lab else None,
                       "silver_labeler_rule": rule_fired(q["query"], "nq_open"),
                       "router_correct": p["temporal"] == q["gold_temporal"],
                       "router_agrees_with_labeler": bool(lab) and p["temporal"] == lab["temporal_class"]})
by_cat = {}
for c in sorted({q["category"] for q in probe_rows}):
    sub = [q for q in probe_rows if q["category"] == c]
    by_cat[c] = {"n": len(sub), "router_temporal_acc": float(np.mean([q["router_correct"] for q in sub])),
                 "router_needs_acc": float(np.mean([q["pred_needs"] == q["gold_needs"] for q in sub])),
                 "silver_labeler_temporal_acc": float(np.mean([q["silver_labeler_temporal"] == q["gold_temporal"] for q in sub])),
                 "router_agrees_with_labeler": float(np.mean([q["router_agrees_with_labeler"] for q in sub]))}
probe_summary = {"n": len(probe_rows),
    "router_temporal_acc": float(np.mean([q["router_correct"] for q in probe_rows])),
    "router_needs_acc": float(np.mean([q["pred_needs"] == q["gold_needs"] for q in probe_rows])),
    "silver_labeler_temporal_acc": float(np.mean([q["silver_labeler_temporal"] == q["gold_temporal"] for q in probe_rows])),
    "router_agrees_with_labeler": float(np.mean([q["router_agrees_with_labeler"] for q in probe_rows])),
    "by_category": by_cat,
    "confusion": metrics([q["gold_temporal"] for q in probe_rows], [q["pred_temporal"] for q in probe_rows],
                         [q["gold_needs"] for q in probe_rows], [q["pred_needs"] for q in probe_rows])}

out = {"meta": {"bundle": "artifacts/router_large_100k/analyzer.pt", "analyzer_sha256": hashlib.sha256(open(BUNDLE/"analyzer.pt","rb").read()).hexdigest(),
                "test_file": "data/router_large/test.jsonl (100k build, n=9941)", "test_sha256": hashlib.sha256(open(ROOT/"data/test_100k.jsonl","rb").read()).hexdigest(),
                "temperature": svc.temperature, "blend_confidence": svc.blend_confidence, "max_length": svc.max_length,
                "device": "cpu (box)", "torch": torch.__version__, "python": platform.python_version(), "inference_seconds": infer_s,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
       "overall_test": overall,
       "matches_kaggle_eval_report": {"kaggle_temporal_accuracy": 0.9987928779800825, "kaggle_temporal_macro_f1": 0.9973338313378042},
       "silver_label_provenance": {"note": "Gold router labels are silver labels produced by src/router_large.label_question (year regex, current/relative-past/when/office regexes, then per-source defaults).",
            "labeler_reproduces_gold_on_test": relabel_match, "n": len(rows),
            "rule_counts": dict(collections.Counter(rules)), "rows_labelled_by_regex_rules": regex_rule_rows,
            "rows_labelled_by_source_default": len(rows) - regex_rule_rows},
       "per_source": per_source, "per_silver_label_rule": per_rule, "label_by_source_counts": per_label_source,
       "source_only_majority_baseline": {"accuracy": source_only_acc, "temporal_macro_f1": src_only_f1, "note": "majority label per source computed on the test set itself (optimistic oracle)"},
       "test_errors": errors,
       "counterfactual_probe": {"summary": probe_summary, "rows": probe_rows}}
json.dump(out, open(ROOT / "learned_router_100k_per_source.json", "w"), indent=2)
print(json.dumps({k: out[k] for k in ["overall_test", "silver_label_provenance", "source_only_majority_baseline"]}, indent=1))
print(json.dumps(probe_summary, indent=1))

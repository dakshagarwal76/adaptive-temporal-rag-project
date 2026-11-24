"""Score a router bundle on probe36 and challenge_v2 (eval-only sets) with the deployed predict()."""
import argparse, json, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("out")
ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
ap.add_argument("--data", default=str(HERE / "data"))
args = ap.parse_args()
sys.path.insert(0, args.repo)
from src.router_large import load_large_router

data = Path(args.data)
svc = load_large_router(args.bundle, device="cpu")
res = {}
sets = (
    ("probe36", json.load(open(data / "probe36.json"))),
    ("challenge_v2", [json.loads(l) for l in open(data / "challenge_v2.jsonl")]),
)
for name, rows in sets:
    det = []
    for r in rows:
        o = svc.predict(r["query"])
        det.append({**r, "pred_temporal": o.temporal_class, "pred_needs": o.needs_retrieval, "confidence": round(o.confidence, 4)})
    tc = sum(d["pred_temporal"] == d["gold_temporal"] for d in det)
    nc = sum(d["pred_needs"] == d["gold_needs"] for d in det)
    bc = sum(d["pred_temporal"] == d["gold_temporal"] and d["pred_needs"] == d["gold_needs"] for d in det)
    cat = collections.defaultdict(lambda: [0, 0])
    for d in det:
        cat[d["category"]][0] += d["pred_temporal"] == d["gold_temporal"]
        cat[d["category"]][1] += 1
    res[name] = {"n": len(det), "temporal_correct": tc, "needs_correct": nc, "both_correct": bc, "by_category_temporal": dict(cat), "rows": det}
    print(name, len(det), "temporal", tc, "needs", nc, "both", bc, dict(cat))
json.dump(res, open(args.out, "w"), indent=1)

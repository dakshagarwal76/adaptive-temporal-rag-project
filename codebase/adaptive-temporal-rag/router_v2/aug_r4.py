"""Append recent-year office-holder counterfactuals to the r4 training split.
Used once after fix_year_labels.py; 274 leak-free pairs were kept in the published run.
"""
import argparse, json, random, sys, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(HERE.parent))
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=29)
    args = ap.parse_args()
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, args.repo)
    import forbidden_index, augment_v2
    from src.router_large import canonicalize

    rng = random.Random(args.seed)
    idx, sets = forbidden_index.build()
    ev = {canonicalize(q) for v in sets.values() for q in v}
    train_path = Path(args.data) / "train.jsonl"
    tr = [json.loads(l) for l in open(train_path)]
    seen = {canonicalize(r["query"]) for r in tr}
    H = [
        "On {d}, which person was serving as {o}'s {h}?",
        "Back in {y}, who was {h} at {o}?",
        "Who served as {h} of {o} during {y}?",
        "Which executive was {h} of {o} on {d}?",
        "During {mon} {y}, who held the {h} role at {o}?",
        "Who was {o}'s {h} at the end of {y}?",
    ]
    C = [
        "On {d}, who is serving as {o}'s {h}?",
        "Who is the {h} of {o} in {mon} 2026?",
        "Who serves as {h} at {o} this {mon}, 2026?",
    ]
    MON = ["January", "March", "May", "July", "September", "November"]
    add, drop = [], 0
    for (h, o) in augment_v2.ORG_OFFICE + augment_v2.COUNTRY_OFFICE:
        for tpl in rng.sample(H, 3):
            y = rng.randint(2019, 2024)
            d = f"{y}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
            add.append((tpl.format(h=h, o=o, y=y, d=d, mon=rng.choice(MON)), "historical"))
        tpl = rng.choice(C)
        d = f"2026-{rng.randint(1,10):02d}-{rng.randint(1,28):02d}"
        add.append((tpl.format(h=h, o=o, d=d, mon=rng.choice(MON)), "current"))
    kept = []
    for q, t in add:
        c = canonicalize(q)
        if c in seen or c in ev or idx.reason(q):
            drop += 1
            continue
        seen.add(c)
        kept.append({
            "query": q, "source": "aug_r4", "temporal_class": t, "needs_retrieval": 1,
            "aug_kind": "recent_year_" + t, "label_confidence": 0.95, "origin": "aug",
            "split": "train", "query_id": "v2_" + hashlib.md5(c.encode()).hexdigest()[:12],
        })
    with open(train_path, "a") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print("generated", len(add), "kept", len(kept), "dropped(leak/dup)", drop)


if __name__ == "__main__":
    main()

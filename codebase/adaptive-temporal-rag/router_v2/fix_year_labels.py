"""Round-4 label fix: Qwen labelled many questions with an explicit past year (2015-2021, mostly
SituatedQA 'as of YEAR') as current. Policy: a real question anchored to an explicit past year
<= reference_year-2 (2024, when today is November 2026) is historical. Years used as quantities
or maths operands are ignored."""
import argparse, json, re, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_v2_dataset import QTY_RE, MATH_RE

YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def anchor_years(q: str):
    m = QTY_RE.sub(" ", q)
    m = MATH_RE.sub(" ", m)
    return [int(y) for y in YEAR.findall(m)]


def apply_year_rule(row: dict, reference_cutoff: int = 2024) -> dict:
    """Return a copy. For real rows, overwrite llm_temporal with historical when warranted."""
    r = dict(row)
    if r.get("origin") == "real":
        ys = anchor_years(r["query"])
        if ys and max(ys) <= reference_cutoff and r.get("llm_temporal") != "historical":
            r["llm_temporal_orig"] = r.get("llm_temporal")
            r["llm_temporal"] = "historical"
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Directory with labelled train/val/test.jsonl")
    ap.add_argument("--dst", required=True, help="Output directory")
    ap.add_argument("--cutoff", type=int, default=2024)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    for split in ("train", "val", "test"):
        out = []
        for line in open(src / f"{split}.jsonl"):
            r = json.loads(line)
            fixed = apply_year_rule(r, args.cutoff)
            if fixed.get("llm_temporal_orig"):
                stats[f"{split}:{fixed['llm_temporal_orig']}->historical"] += 1
            out.append(fixed)
        with open(dst / f"{split}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    print(dict(stats))


if __name__ == "__main__":
    main()

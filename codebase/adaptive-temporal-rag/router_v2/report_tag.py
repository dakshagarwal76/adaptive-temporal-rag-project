"""report_tag.py [--results DIR] <tag> <dataset>: print measured summary + significance vs frozen adaptive."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("dataset")
ap.add_argument("--results", default=str(Path(__file__).resolve().parent / "results"))
args = ap.parse_args()
root = Path(args.results)
s = json.load(open(root / f"{args.tag}_{args.dataset}_significance.json"))[args.dataset]
for k, v in s.items():
    for m in ("em", "contains", "token_f1"):
        if m in v:
            x = v[m]
            print(
                f"{args.tag} {args.dataset} {k} {m}: {x['mean_a']:.3f} vs {x['mean_b']:.3f} "
                f"diff {x['diff_a_minus_b']:+.3f} CI[{x['ci95_low']:.3f},{x['ci95_high']:.3f}] "
                f"a_only={x['n_a_only']} b_only={x['n_b_only']} p={x['mcnemar_p']:.4g}"
            )

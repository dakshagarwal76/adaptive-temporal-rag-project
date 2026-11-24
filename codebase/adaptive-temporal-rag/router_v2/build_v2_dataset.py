"""Build router_v2 data: relabelled 100k real rows + counterfactual augmentation, leakage-screened.
v2 label schema (silver_v2; the Colab notebook can overwrite temporal labels with LLM labels):
  needs_retrieval = 0 only for self-contained questions (arithmetic, unit/calendar facts, word trivia)
                    that the extractive reader can answer without evidence; all real QA rows -> 1.
  temporal_class  = regex labeller, but year-like numbers used as quantities are not time anchors and
                    metalinguistic uses of time words are not current cues."""
import json, re, random, sys, collections, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(__import__('os').environ.get('ATRAG_REPO', HERE.parent))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(HERE))
from src.router_large import label_question, canonicalize
import forbidden_index, augment_v2

SRC = Path(__import__('os').environ.get('ROUTER_V2_SRC100K', HERE / 'src_100k' / 'data' / 'router_large'))
OUT = Path(__import__('os').environ.get('ROUTER_V2_DATA', HERE / 'data')); OUT.mkdir(exist_ok=True)
UNIT = r"(?:seconds?|minutes?|hours?|days?|weeks?|grams?|kilograms?|kg|km|metres?|meters?|miles?|feet|inches|litres?|liters?|calories|dollars?|pounds?|euros?|people|units|points|pages|words|bc|bce)"
QTY_RE = re.compile(rf"\b(1[0-9]{{3}}|20[0-9]{{2}})\s+{UNIT}\b", re.I)
MATH_RE = re.compile(r"(?:(?:square root of|cube root of|plus|minus|times|divided by|multiplied by|sum of|product of)\s+(1[0-9]{3}|20[0-9]{2})\b)|(?:\bis\s+(1[0-9]{3}|20[0-9]{2})\s+(?:a prime|an? (?:even|odd|prime)|divisible))", re.I)
META_RE = re.compile(r"\b(?:the word|the term|the adverb|the adjective|means?|meant by)\b.{0,25}\b(?:today|now|current|latest|recent|present)\b|most recent common ancestor", re.I)

def label_v2(q, source, old):
    masked = QTY_RE.sub(lambda m: 'NUMQTY ' + m.group(0).split(None, 1)[1], q)
    masked = MATH_RE.sub(lambda m: m.group(0).replace(m.group(1) or m.group(2), 'NUMQTY'), masked)
    lab = label_question(masked, source) if masked != q else None
    t = (lab or {}).get('temporal_class', old['temporal_class']) if masked != q else old['temporal_class']
    if META_RE.search(q) and t == 'current':
        t = 'timeless'
    return t

def main(seed=13):
    rng = random.Random(seed)
    idx, sets = forbidden_index.build()
    report = {'leak_dropped': collections.Counter(), 'relabel': collections.Counter()}
    rows = {}
    eval_exact = {canonicalize(q) for k, v in sets.items() for q in v}
    for split in ('train', 'val', 'test'):
        out = []
        for l in open(SRC / f'{split}.jsonl'):
            r = json.loads(l)
            why = 'exact_any_split' if canonicalize(r['query']) in eval_exact else idx.reason(r['query'], entity=r.get('entity') or None)
            if why:
                report['leak_dropped'][f'real_{split}_{why}'] += 1; continue
            t2 = label_v2(r['query'], r['source'], r)
            report['relabel'][f"{r['temporal_class']}->{t2}"] += 1
            out.append({'query': r['query'], 'source': r['source'], 'temporal_class': t2, 'needs_retrieval': 1,
                        'old_temporal': r['temporal_class'], 'old_needs': r['needs_retrieval'],
                        'label_confidence': r.get('label_confidence', 0.8), 'origin': 'real', 'split': split})
        rows[split] = out
    aug = augment_v2.gen(random.Random(seed))
    seen = set(); kept = []
    for r in aug:
        c = canonicalize(r['query'])
        if c in seen: report['leak_dropped']['aug_dup'] += 1; continue
        why = 'exact_any_split' if c in eval_exact else idx.reason(r['query'])
        if why: report['leak_dropped'][f'aug_{why}'] += 1; continue
        seen.add(c); kept.append(r)
    rng.shuffle(kept)
    n = len(kept); a_val = kept[: n // 10]; a_test = kept[n // 10: n // 5]; a_train = kept[n // 5:]
    for split, part in (('train', a_train), ('val', a_val), ('test', a_test)):
        for r in part:
            rows[split].append({'query': r['query'], 'source': 'aug_v2', 'temporal_class': r['temporal_class'],
                                'needs_retrieval': r['needs_retrieval'], 'aug_kind': r['aug_kind'],
                                'label_confidence': 0.95, 'origin': 'aug', 'split': split})
    # cross-split dedupe (val/test rows whose canonical form appears in train are dropped)
    train_keys = {canonicalize(r['query']) for r in rows['train']}
    for split in ('val', 'test'):
        before = len(rows[split]); rows[split] = [r for r in rows[split] if canonicalize(r['query']) not in train_keys]
        report['leak_dropped'][f'{split}_overlap_train'] += before - len(rows[split])
    for split, part in rows.items():
        for r in part:
            r['query_id'] = 'v2_' + hashlib.md5(canonicalize(r['query']).encode()).hexdigest()[:12]
        with open(OUT / f'{split}.jsonl', 'w') as f:
            for r in part: f.write(json.dumps(r) + '\n')
    # final leakage verification on every written row
    final = collections.Counter()
    eval_exact = {canonicalize(q) for k, v in sets.items() for q in v}
    for split, part in rows.items():
        for r in part:
            if canonicalize(r['query']) in eval_exact: final[f'{split}_exact_eval_overlap'] += 1
            if idx.reason(r['query']): final[f'{split}_index_hit'] += 1
    stats = {
        'n': {s: len(p) for s, p in rows.items()},
        'by_split_class': {s: dict(collections.Counter(f"{r['temporal_class']}|{r['needs_retrieval']}" for r in p).most_common()) for s, p in rows.items()},
        'by_split_origin': {s: dict(collections.Counter(r['origin'] for r in p)) for s, p in rows.items()},
        'aug_generated': len(aug), 'aug_kept': len(kept),
        'leak_dropped': dict(report['leak_dropped']), 'relabel_transitions_real': dict(report['relabel']),
        'final_leak_check': dict(final), 'eval_set_sizes': {k: len(v) for k, v in sets.items()},
    }
    stats = json.loads(json.dumps(stats, default=str))
    (OUT / 'stats_v2.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))

if __name__ == '__main__':
    main()

"""router_v2 trainer. Writes a bundle loadable by src.router_large.load_large_router
(analyzer.pt = MultitaskQueryAnalyzer state_dict, analyzer_meta.json).
Changes vs v1: v2 labels (optionally LLM temporal labels), class-balanced sampling over
(temporal, needs) cells, and a confidence head trained with BCE on whether both heads are
correct (v1 trained it toward silver label_confidence)."""
import argparse, json, math, os, random, sys, time
from collections import Counter
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--data', default=str(Path(__file__).resolve().parent / 'data'))
    ap.add_argument('--out', default=str(Path(__file__).resolve().parent / 'bundles' / 'run'))
    ap.add_argument('--model', default='microsoft/deberta-v3-small')
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--max-len', type=int, default=64)
    ap.add_argument('--train-cap', type=int, default=0, help='subsample real train rows (0 = all)')
    ap.add_argument('--use-llm-labels', action='store_true')
    ap.add_argument('--conf-weight', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--val-cap', type=int, default=0)
    ap.add_argument('--test-cap', type=int, default=0)
    args = ap.parse_args()
    sys.path.insert(0, args.repo)
    from src.query_analyzer import MultitaskQueryAnalyzer, LABEL2ID, ID2LABEL, TEMPORAL_LABELS
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    if args.threads: torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def load(split):
        rows = [json.loads(l) for l in open(Path(args.data) / f'{split}.jsonl')]
        if args.use_llm_labels:
            for r in rows:
                if r.get('origin') == 'real' and r.get('llm_temporal') in TEMPORAL_LABELS:
                    r['silver_temporal'] = r['temporal_class']; r['temporal_class'] = r['llm_temporal']
        return rows
    train, val, test = load('train'), load('val'), load('test')
    if args.train_cap:
        rng = random.Random(args.seed)
        real = [r for r in train if r['origin'] == 'real']; aug = [r for r in train if r['origin'] != 'real']
        cur = [r for r in real if r['temporal_class'] == 'current']; rest = [r for r in real if r['temporal_class'] != 'current']
        train = aug + cur + rng.sample(rest, max(0, args.train_cap - len(cur)))
        val = [r for r in val if r['origin'] != 'real'] + random.Random(1).sample([r for r in val if r['origin'] == 'real'], min(3000, len(val)))
    if args.val_cap: val = random.Random(2).sample(val, min(args.val_cap, len(val)))
    if args.test_cap: test = random.Random(3).sample(test, min(args.test_cap, len(test)))
    cell = Counter((r['temporal_class'], r['needs_retrieval']) for r in train)
    print('train cells', dict(cell), 'n', len(train), flush=True)
    weights = [1.0 / math.sqrt(cell[(r['temporal_class'], r['needs_retrieval'])]) for r in train]

    tok = AutoTokenizer.from_pretrained(args.model)
    def collate(batch):
        enc = tok([r['query'] for r in batch], truncation=True, padding='max_length', max_length=args.max_len, return_tensors='pt')
        enc['needs'] = torch.tensor([int(r['needs_retrieval']) for r in batch])
        enc['temporal'] = torch.tensor([LABEL2ID[r['temporal_class']] for r in batch])
        return enc
    steps_per_epoch = math.ceil(len(train) / args.bs)
    total = int(steps_per_epoch * args.epochs)
    sampler = WeightedRandomSampler(weights, num_samples=total * args.bs, replacement=True, generator=torch.Generator().manual_seed(args.seed))
    dl = DataLoader(train, batch_size=args.bs, sampler=sampler, collate_fn=collate, num_workers=2 if dev == 'cuda' else 0)
    model = MultitaskQueryAnalyzer(model_name=args.model).float().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    bce = nn.BCELoss()

    @torch.no_grad()
    def predict(rows, T=1.0):
        model.eval(); res = []
        for i in range(0, len(rows), 128):
            b = rows[i:i + 128]
            enc = tok([r['query'] for r in b], truncation=True, padding='max_length', max_length=args.max_len, return_tensors='pt').to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=args.bf16 and dev == 'cuda'):
                o = model(enc['input_ids'], enc['attention_mask'])
            res.append((o['necessity_logits'].float().cpu(), o['temporal_logits'].float().cpu(), o['confidence'].float().cpu()))
        model.train()
        return [torch.cat(x) for x in zip(*res)]

    def macro_f1(gold, pred, labels):
        fs = []
        for l in labels:
            tp = sum(g == l and p == l for g, p in zip(gold, pred)); fp = sum(g != l and p == l for g, p in zip(gold, pred)); fn = sum(g == l and p != l for g, p in zip(gold, pred))
            if tp + fn == 0: continue
            pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn); fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        return sum(fs) / len(fs)

    def ece(conf, correct, bins=15):
        e, n = 0.0, len(conf)
        for b in range(bins):
            lo, hi = b / bins, (b + 1) / bins
            idx = [i for i, c in enumerate(conf) if (c > lo or (b == 0 and c >= lo)) and c <= hi]
            if idx: e += len(idx) / n * abs(sum(correct[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
        return e

    def score(rows, T, gold_t='temporal_class', gold_n='needs_retrieval'):
        nl, tl, hc = predict(rows)
        npb = torch.softmax(nl / T, -1); tpb = torch.softmax(tl / T, -1)
        np_, tp_ = npb.argmax(-1).tolist(), tpb.argmax(-1).tolist()
        gt = [LABEL2ID[r[gold_t]] for r in rows]; gn = [int(r[gold_n]) for r in rows]
        tc = [int(a == b) for a, b in zip(tp_, gt)]; nc = [int(a == b) for a, b in zip(np_, gn)]
        both = [a * b for a, b in zip(tc, nc)]
        blend = [0.5 * h + 0.5 * max(a, b) for h, a, b in zip(hc.tolist(), npb.max(-1).values.tolist(), tpb.max(-1).values.tolist())]
        return {'n': len(rows), 'temporal_acc': sum(tc) / len(rows), 'needs_acc': sum(nc) / len(rows),
                'temporal_macro_f1': macro_f1(gt, tp_, [0, 1, 2]), 'needs_macro_f1': macro_f1(gn, np_, [0, 1]),
                'n_temporal_correct': sum(tc), 'n_needs_correct': sum(nc),
                'ece_temporal': ece(tpb.max(-1).values.tolist(), tc), 'ece_needs': ece(npb.max(-1).values.tolist(), nc),
                'ece_conf_head_vs_both_correct': ece(hc.tolist(), both), 'ece_blended_conf_vs_both_correct': ece(blend, both)}, (np_, tp_, hc.tolist())

    t0 = time.time(); model.train(); step = 0; best = -1; hist = []
    eval_every = max(200, total // 6)
    for batch in dl:
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=args.bf16 and dev == 'cuda'):
            o = model(batch['input_ids'], batch['attention_mask'], needs=batch['needs'], temporal=batch['temporal'])
        with torch.no_grad():
            correct = ((o['necessity_logits'].argmax(-1) == batch['needs']) & (o['temporal_logits'].argmax(-1) == batch['temporal'])).float()
        loss = o['loss'] + args.conf_weight * bce(o['confidence'].float().clamp(1e-6, 1 - 1e-6), correct)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
        if step % 50 == 0: print(f'step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
        if step % eval_every == 0 or step == total:
            m, _ = score(val, 1.0); f = 0.5 * (m['temporal_macro_f1'] + m['needs_macro_f1']); hist.append({'step': step, **m})
            print('VAL', step, json.dumps(m), flush=True)
            if f > best:
                best = f; torch.save(model.state_dict(), out / 'analyzer.pt')
        if step >= total: break
    model.load_state_dict(torch.load(out / 'analyzer.pt', map_location='cpu'))
    # temperature on val temporal logits (same grid as v1)
    _, tl, _ = predict(val); gt = torch.tensor([LABEL2ID[r['temporal_class']] for r in val])
    T = min((s / 10 for s in range(5, 31)), key=lambda t: float(nn.functional.cross_entropy(tl / t, gt)))
    meta = {'model_name': args.model, 'tokenizer_name': args.model, 'max_length': args.max_len, 'conf_threshold': 0.55,
            'temporal_labels': list(TEMPORAL_LABELS), 'temperature': T, 'temperature_fitted': True, 'blend_confidence': True,
            'confidence_loss': 'bce_on_both_heads_correct', 'confidence_loss_weight': args.conf_weight,
            'label_schema': 'v2' + ('+llm_temporal' if args.use_llm_labels else ''), 'train_n': len(train),
            'train_cells': {f'{a}|{b}': c for (a, b), c in cell.items()}, 'epochs': args.epochs, 'lr': args.lr, 'bs': args.bs,
            'seed': args.seed, 'device': dev, 'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
            'train_seconds': time.time() - t0, 'history': hist}
    (out / 'analyzer_meta.json').write_text(json.dumps(meta, indent=2))
    probe = [json.loads(x) for x in [json.dumps(r) for r in json.load(open(Path(args.data) / 'probe36.json'))]]
    chal = [json.loads(l) for l in open(Path(args.data) / 'challenge_v2.jsonl')]
    rep = {'temperature': T, 'val': score(val, T)[0], 'test': score(test, T)[0],
           'test_real_only': score([r for r in test if r['origin'] == 'real'], T)[0],
           'test_aug_only': score([r for r in test if r['origin'] != 'real'], T)[0]}
    for name, rows in (('probe36', probe), ('challenge_v2', chal)):
        m, (np_, tp_, hc) = score(rows, T, 'gold_temporal', 'gold_needs')
        rep[name] = m
        rep[name + '_rows'] = [{'query': r['query'], 'category': r['category'], 'gold_temporal': r['gold_temporal'], 'gold_needs': r['gold_needs'],
                                'pred_temporal': ID2LABEL[t], 'pred_needs': n, 'conf_head': round(h, 4)} for r, n, t, h in zip(rows, np_, tp_, hc)]
    (out / 'eval_report_v2.json').write_text(json.dumps(rep, indent=2))
    print('FINAL', json.dumps({k: v for k, v in rep.items() if not k.endswith('_rows')}, indent=1), flush=True)


if __name__ == "__main__":
    main()

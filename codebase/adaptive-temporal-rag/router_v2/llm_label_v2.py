"""LLM temporal relabelling for router_v2 (run on an A100/L4 GPU).
Scores the next-token logits of the three label words after a rubric prompt with
Qwen2.5-7B-Instruct (no sampling), and writes llm_temporal / llm_margin into each row.
Only the router_v2 train/val/test files are labelled; eval sets are never passed in."""
import argparse, json, time, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True)
ap.add_argument('--model', default='Qwen/Qwen2.5-7B-Instruct'); ap.add_argument('--bs', type=int, default=96)
ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
SYS = ("You label questions for a time-aware retrieval system. Today is November 2026. Choose exactly one label:\n"
       "current - the correct answer depends on the present moment and may change over time: who holds an office or title now, "
       "latest or most recent results, present-day statistics, prices or rankings. This applies even when the question has no word like 'current' or 'now'.\n"
       "historical - the question is anchored to a specific past time or past event: an explicit past year or date, 'before/after X', 'the first ...', "
       "'during the war', 'when did ... happen', or who won/did something at a past occasion.\n"
       "timeless - the answer does not depend on when the question is asked: definitions, science, mathematics, geography, language, "
       "and stable facts such as who wrote a book. Numbers that only look like years (for example '2048 bytes' or 'is 1729 a cube number') "
       "are not time anchors, and talking about a time word (for example 'what does the word yesterday mean') is timeless.\n"
       "Reply with one word: current, historical, or timeless.")
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map='cuda').eval()
labels = ['current', 'historical', 'timeless']
ids = [tok.encode(w, add_special_tokens=False)[0] for w in labels]
assert len(set(ids)) == 3, ids
def prompt(q):
    return tok.apply_chat_template([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f'Question: {q}\nLabel:'}],
                                   tokenize=False, add_generation_prompt=True)
for split in ('train', 'val', 'test'):
    p = Path(a.data) / f'{split}.jsonl'; rows = [json.loads(l) for l in open(p)]
    todo = list(range(len(rows)))[: a.limit or None]; t0 = time.time()
    for i in range(0, len(todo), a.bs):
        b = todo[i:i + a.bs]
        enc = tok([prompt(rows[j]['query']) for j in b], return_tensors='pt', padding=True).to('cuda')
        with torch.inference_mode():
            logits = model(**enc).logits[:, -1, :].float()
        lp = torch.log_softmax(logits[:, ids], -1)
        top = lp.topk(2, -1)
        for k, j in enumerate(b):
            rows[j]['llm_temporal'] = labels[int(top.indices[k, 0])]
            rows[j]['llm_margin'] = float(top.values[k, 0] - top.values[k, 1])
        if (i // a.bs) % 50 == 0: print(split, i, len(todo), f'{time.time() - t0:.0f}s', flush=True)
    with open(p, 'w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    lab = [r for r in rows if 'llm_temporal' in r]
    agree = sum(r['llm_temporal'] == r['temporal_class'] for r in lab)
    print(f'{split}: labelled {len(lab)} agree_with_silver_v2 {agree}', flush=True)

"""Build the v2 leakage index: original eval test sets + the 36-question counterfactual probe
+ the v2 challenge set. Nothing in these files may enter router_v2 training."""
import json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('ATRAG_REPO', HERE.parent))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/'scripts'))
from src.router_large import LeakageIndex, canonicalize
import build_large_router_dataset as b

PROBE = Path(os.environ.get('PROBE36', HERE / 'data' / 'probe36.json'))
CHALLENGE = Path(os.environ.get('CHALLENGE_V2', HERE / 'data' / 'challenge_v2.jsonl'))

PROBE_CHALLENGE_ENTITIES = [
 # probe36
 'federal reserve','microsoft','bitcoin','nba','manchester united','fifa','wimbledon','kofi annan','tim cook',
 'berlin wall','cuban missile crisis','west germany','barack obama','beijing','cold war','carbon 14','leap year',
 'leap years','square root','most recent common ancestor','solar system','nobel prize in literature','super bowl',
 'german federal election','inflation rate','population of india','japan','japans prime minister',
 # challenge_v2
 'european space agency','world health organization','unemployment rate','samsung','governor general',
 'exchange rate','real madrid','bundesliga','world chess champion','nigeria','mayor of paris',
 'international monetary fund','gorbachev','eurovision','satya nadella','apollo 11','tour de france','brasilia',
 'battle of waterloo','suez canal','london games','disney','angela merkel','benedict xvi','ballon dor',
 'palme dor','canadian prime minister','booker prize','stanley cup','nato','rugby world cup','winter olympics',
 'leap second','nowadays','mitosis','ocean trench','radiocarbon','middlemarch','tungsten','photosphere',
 'binary search tree','pancreas','rite of spring','neon','night watch','spider','triangle']

def eval_question_sets():
    sets = {}
    for name, p in [('synthetic', REPO/'data/queries.jsonl'), ('heldout', REPO/'data/heldout/queries.jsonl'),
                    ('timeqa', REPO/'data/general/timeqa/queries.jsonl')]:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        sets[name+'_test'] = [r['query'] for r in rows if r.get('split') == 'test']
        sets[name+'_all'] = [r['query'] for r in rows]
    sets['probe36'] = [r['query'] for r in json.load(open(PROBE))]
    if CHALLENGE.exists():
        sets['challenge_v2'] = [json.loads(l)['query'] for l in open(CHALLENGE) if l.strip()]
    return sets

def build():
    idx = b.build_forbidden_index()
    sets = eval_question_sets()
    extra = sets['probe36'] + sets.get('challenge_v2', []) + sets['synthetic_all']
    idx.add_entities(PROBE_CHALLENGE_ENTITIES)
    idx.add_exact(extra)
    idx.add_question_templates(extra)
    return idx, sets

if __name__ == '__main__':
    idx, sets = build()
    d = idx.to_dict()
    print({k: len(v) for k, v in d.items()})
    Path('forbidden_v2.json').write_text(json.dumps(d))
    print(sorted(d['forbidden_entities'])[:400])

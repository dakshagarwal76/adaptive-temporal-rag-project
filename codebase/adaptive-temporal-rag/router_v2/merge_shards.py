"""merge_shards.py <shard_root> <n> <dataset> <final_out_dir>: interleave shard CSVs back into original query order."""
import sys, pandas as pd
from pathlib import Path
root, n, ds, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys_name = 'adaptive_learned100k'
parts = [pd.read_csv(root / str(k) / ds / f'results_{sys_name}.csv') for k in range(n)]
rows = []
for j in range(max(len(p) for p in parts)):
    for k in range(n):
        if j < len(parts[k]): rows.append(parts[k].iloc[j])
df = pd.DataFrame(rows).reset_index(drop=True)
(out / ds).mkdir(parents=True, exist_ok=True)
df.to_csv(out / ds / f'results_{sys_name}.csv', index=False, encoding='utf-8')
print('merged', ds, len(df), 'rows')

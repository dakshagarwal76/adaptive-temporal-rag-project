#!/usr/bin/env bash
# usage: run_eval_pipeline.sh <abs bundle dir> <tag>
# Optional env: ATRAG_REPO, ROUTER_V2_ROOT, NSYN, NHELD, NTQA, DATASETS, SKIP_PROBES
set -u
B="$1"
T="$2"
RV="${ROUTER_V2_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
REPO="${ATRAG_REPO:-$(cd "$RV/.." && pwd)}"
PY="${PYTHON:-python}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false ROUTER_HYBRID=0
cd "$RV"
L=$RV/results/pipe_$T
mkdir -p "$L"
date +"%T start" > "$L/status"
[ "${SKIP_PROBES:-0}" = 1 ] || (cd "$RV" && $PY score_probes.py "$B" results/probes_$T.json > "$L/probes.log" 2>&1; echo "probes done" >> "$L/status") &
declare -A N=([synthetic]=${NSYN:-2} [heldout]=${NHELD:-2} [timeqa]=${NTQA:-4})
for ds in ${DATASETS:-synthetic heldout timeqa}; do
  (
    n=${N[$ds]}
    SR=$RV/results/general_eval/${T}_shards/$ds
    pids=()
    for k in $(seq 0 $((n-1))); do
      (cd "$REPO" && ROUTER_TAG=${T}_${ds}_s$k ROUTER_SHARD=$k/$n ATRAG_REPO="$REPO" ROUTER_V2_RESULTS="$RV/results" \
         $PY "$RV/eval_router_v2.py" --router-dir "$B" --datasets $ds \
         --systems adaptive_learned100k --out-dir $SR/$k > "$L/${ds}_s$k.log" 2>&1) & pids+=($!)
    done
    wait "${pids[@]}"
    FO=$RV/results/general_eval/${T}_$ds
    $PY "$RV/merge_shards.py" $SR $n $ds $FO >> "$L/${ds}_merge.log" 2>&1
    (cd "$REPO" && ROUTER_TAG=${T}_$ds ATRAG_REPO="$REPO" ROUTER_V2_RESULTS="$RV/results" \
       $PY "$RV/eval_router_v2.py" --router-dir "$B" --datasets $ds --systems adaptive_learned100k \
       --out-dir $FO --stats-only >> "$L/${ds}_merge.log" 2>&1)
    $PY "$RV/report_tag.py" ${T} $ds --results "$RV/results" > "$L/${ds}_report.txt" 2>&1
    echo "$ds done $(date +%T)" >> "$L/status"
  ) &
done
wait
echo "all done $(date +%T)" >> "$L/status"

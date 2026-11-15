#!/bin/bash
# Reruns every script of Chapter 55 (run from anywhere).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in folds_hand strat_scratch sk_splitters sk_validate err_groups case_groups case_series kbias nested prose_numbers; do
  python3 $W code/$f.py > verify/$f.out 2> verify/$f.err || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py; python3 fig/probs.py > /dev/null

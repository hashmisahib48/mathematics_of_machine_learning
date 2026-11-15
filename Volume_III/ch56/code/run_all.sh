#!/bin/bash
# Reruns every script of Chapter 56 (run from anywhere); about five minutes in all.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
# pool.py and winner.json feed the figures, case_search.py feeds figures and problems
for f in halving_hand grid_count grid_scratch sk_grid sk_other sk_models err_names pool winner case_search prose_numbers; do
  python3 $W code/$f.py > verify/$f.out 2> verify/$f.err || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py > /dev/null; python3 $W fig/probs.py > /dev/null

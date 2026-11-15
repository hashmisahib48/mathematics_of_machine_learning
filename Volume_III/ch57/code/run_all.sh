#!/bin/bash
# Reruns every script of Chapter 57 (run from anywhere).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in screen_story confusion_hand roc_hand average_hand scratch sk_classification sk_regression scorers sk_other err_scores case_cancer prose_numbers; do
  python3 $W code/$f.py > verify/$f.out 2> verify/$f.err || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py > /dev/null; python3 fig/probs.py > /dev/null

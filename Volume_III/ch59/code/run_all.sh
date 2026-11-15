#!/bin/bash
# Reruns every script of Chapter 59 (run from anywhere).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in audit case_a1 case_a2 case_b1 case_b2 case_c save_model repeats hand; do
  python3 $W code/$f.py > verify/$f.out 2> verify/$f.err || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py; python3 fig/probs.py > /dev/null

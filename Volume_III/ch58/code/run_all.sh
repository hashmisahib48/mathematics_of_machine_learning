#!/bin/bash
# Reruns every script of Chapter 58 (run from anywhere).
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in story hand scratch thr_scratch sk case prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py; python3 fig/probs.py > /dev/null

#!/bin/bash
# Reruns every script of the Chapter 16 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s16b1 s16b2 s16b3 c1 c2 c3; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

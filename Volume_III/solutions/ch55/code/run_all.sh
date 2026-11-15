#!/bin/bash
# Reruns every script of the Chapter 55 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers q_rep q_series c1 s55b1 s55b2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c2"; python3 $W ../fig/c2.py > ../verify/c2.out
echo "== eqs"; python3 $W eqs.py

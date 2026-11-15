#!/bin/bash
# Reruns every script of the Chapter 56 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_hit q_grid q_rungs c2 s56b1 s56b2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c1"; python3 $W ../fig/c1.py > ../verify/c1.out
echo "== eqs"; python3 $W eqs.py

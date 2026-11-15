#!/bin/bash
# Reruns every script of the Chapter 59 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s59b1; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
for s in c1 c2 c3; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

#!/bin/bash
# Reruns every script of the Chapter 31 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s31b1 s31b2 s31b3; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out 2>&1
done
for s in c1 c2 c3; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out 2>&1
done
echo "== eqs"; python3 $W eqs.py

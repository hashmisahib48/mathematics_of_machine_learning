#!/bin/bash
# Reruns every script of the Chapter 18 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s18b1 s18b2 s18b3 c2_numbers c3_numbers; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c1"; python3 $W ../fig/c1.py > ../verify/c1.out
echo "== eqs"; python3 $W eqs.py

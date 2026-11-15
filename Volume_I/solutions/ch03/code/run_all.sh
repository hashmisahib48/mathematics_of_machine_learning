#!/bin/bash
# Reruns every script of the Chapter 3 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s3b1 s3b2 s3b3 c_numbers; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
for s in c3; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

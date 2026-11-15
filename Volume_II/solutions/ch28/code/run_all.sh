#!/bin/bash
# Reruns every script of the Chapter 28 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s28b1 s28b2 c1; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
for s in c2 c3; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

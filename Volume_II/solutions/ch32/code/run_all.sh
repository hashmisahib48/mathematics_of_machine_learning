#!/bin/bash
# Reruns every script of the Chapter 32 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s32b1 s32b2 s32b3 c3; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
for s in c1 c2; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

#!/bin/bash
# Reruns every script of the Chapter 10 solutions and regenerates outputs and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers c2_numbers c3_numbers s10b1 s10b2 s10b3; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

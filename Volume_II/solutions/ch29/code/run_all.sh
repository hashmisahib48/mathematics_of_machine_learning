#!/bin/bash
# Reruns every script of the Chapter 29 solutions and regenerates outputs and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers q_numbers2 s29b1 s29b2 s29b3 c1 c2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

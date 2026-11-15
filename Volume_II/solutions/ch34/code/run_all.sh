#!/bin/bash
# Reruns every script of the Chapter 34 solutions and regenerates outputs, figures and equation images.
set -e
export OMP_NUM_THREADS=1
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s34b1 s34b3 c1 c2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== s34b2 (2 threads: the result depends on the thread count, see threads.sh)"
OMP_NUM_THREADS=2 python3 $W s34b2.py > ../verify/s34b2.out
echo "== threads"; bash threads.sh > ../verify/s34b2_threads.out
echo "== fig/c3"; python3 $W ../fig/c3.py > ../verify/c3.out
echo "== eqs"; python3 $W eqs.py

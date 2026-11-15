#!/bin/bash
# Reruns every script of the Chapter 9 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
W="-W error::FutureWarning -W error::DeprecationWarning"
cp ../../../ch09/code/ols_scratch.py .          # the class of Listing 9.2, imported by s9b1.py and s9b3.py
for s in hand q_numbers c2_numbers c3_numbers s9b1 s9b2 s9b3; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c2"; python3 $W ../fig/c2.py > ../verify/c2fig.out
echo "== eqs"; python3 $W eqs.py

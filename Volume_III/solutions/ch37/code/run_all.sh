#!/bin/bash
# Reruns every script of the Chapter 37 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s37b1 s37b2 c1_numbers; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
mv ../verify/c1_numbers.out ../verify/c1.out
echo "== fig/c2"; python3 $W ../fig/c2.py > ../verify/c2.out
echo "== eqs"; python3 $W eqs.py

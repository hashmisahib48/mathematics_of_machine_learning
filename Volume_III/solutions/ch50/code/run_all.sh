#!/bin/bash
# Reruns every script of the Chapter 50 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s50b1 s50b2 extra c1; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c2"; python3 $W ../fig/c2.py > ../verify/c2.out
echo "== eqs"; python3 $W eqs.py

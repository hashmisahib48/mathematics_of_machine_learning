#!/bin/bash
# Reruns every script of the Chapter 49 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers s49b1 s49b2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
for s in c1 c2; do
  echo "== fig/$s"; python3 $W ../fig/$s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

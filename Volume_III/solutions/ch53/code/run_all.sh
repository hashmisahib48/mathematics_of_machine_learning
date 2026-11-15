#!/bin/bash
# Reruns every script of the Chapter 53 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers c2 c3 s53b1 s53b2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== fig/c1"; python3 $W ../fig/c1.py > ../verify/c1.out
echo "== eqs"; python3 $W eqs.py

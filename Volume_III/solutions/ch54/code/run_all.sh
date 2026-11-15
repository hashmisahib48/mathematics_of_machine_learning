#!/bin/bash
# Reruns every script of the Chapter 54 solutions and regenerates outputs and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers c1 c2 s54b1 s54b2; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
echo "== eqs"; python3 $W eqs.py

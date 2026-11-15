#!/bin/bash
# Reruns every script of the Chapter 39 solutions and regenerates outputs, figures and equation images.
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in hand q_numbers c1 s39b1 s39b2 edges; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out 2>/dev/null
done
echo "== fig/c2"; python3 $W ../fig/c2.py > ../verify/c2.out
echo "== eqs"; python3 $W eqs.py

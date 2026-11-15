#!/bin/bash
# Reruns every script of Chapter 36 and refreshes verify/*.out and verify/*.json (about 2 minutes in all).
set -e
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in toy toy_more kmeans_check sklearn_toy variants knobs case_wine case_digits quantise; do
  echo "== $s"; python3 $W $s.py > ../verify/$s.out
done
cd ../fig
for s in figs figs2 figs3 eqs probs; do
  echo "== fig/$s"; python3 $W $s.py
done

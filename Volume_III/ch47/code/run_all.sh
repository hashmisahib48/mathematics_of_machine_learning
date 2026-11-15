#!/bin/sh
# Reruns every script of Chapter 47 (from the chapter folder or anywhere).
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in toy kde_scratch kde_gauss kde_sklearn select_h case_iris case_digits repeats prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
for f in eqs figs1 figs2 probs; do python3 $W fig/$f.py > /dev/null || exit 1; done
echo done

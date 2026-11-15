#!/bin/sh
# Reruns every script of Chapter 42; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in toy toy_more pca_scratch pca_sklearn solvers kpca_toy kpca_circles incr_svd sparse_pca case_digits case_wine; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 $W code/prose_numbers.py > verify/prose_numbers.out || exit 1
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null

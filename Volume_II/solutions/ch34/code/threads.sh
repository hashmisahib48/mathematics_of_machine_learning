#!/bin/bash
# Problem 34B.2 depends on the number of BLAS threads: the propagation has not converged at max_iter = 1000.
cd "$(dirname "$0")"
for n in 1 2 4 8; do
  echo "threads $n:"; OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n python3 s34b2.py 2>/dev/null | grep -E "accuracy|^propagation|^spreading"
done

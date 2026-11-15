#!/bin/sh
# Regenerates every file in verify/*.out from the programs in code/.
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
for f in toy sweep svr_scratch svr_sklearn nusvr linear_svr compare_krr checks case_diabetes case_wrecked; do
  python3 -W error::FutureWarning -W error::DeprecationWarning code/$f.py > verify/$f.out 2>&1
done

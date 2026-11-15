#!/bin/sh
# Regenerates every file in verify/*.out from the programs in code/.
cd "$(dirname "$0")/.."
for f in toy csweep smo_hand smo_scratch svc_sklearn multiclass class_weight nusvc linear_svc calibrate warn loo_bound cost checks case_cancer; do
  python3 -W error::FutureWarning -W error::DeprecationWarning code/$f.py > verify/$f.out 2>&1
done

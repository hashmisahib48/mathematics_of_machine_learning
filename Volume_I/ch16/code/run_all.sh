#!/bin/sh
# Regenerates every file in verify/*.out from the programs in code/.
cd "$(dirname "$0")/.."
for f in toy objective_check penalty_api lr_scratch softmax_check lr_sklearn choose_C case_cancer case_cancer2 separation checks prose_numbers; do
  python3 code/$f.py > verify/$f.out 2>&1
done

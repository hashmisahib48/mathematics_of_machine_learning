#!/bin/bash
# Reruns every script of Chapter 37 and captures the listing outputs.
# Run from anywhere: bash ch37/code/run_all.sh
set -e
cd "$(dirname "$0")/.."
W="-W error::FutureWarning -W error::DeprecationWarning"
mkdir -p verify
for f in toy toy_tables agglo_test agglo_sklearn errors birch_cf case_wine case_digits prose_numbers timing; do
  python3 $W code/$f.py > verify/$f.out 2>&1
  echo "ran $f"
done
python3 $W fig/eqs.py
python3 $W fig/figs.py > verify/figs.out 2>&1
python3 $W fig/probs.py > verify/probs.out 2>&1
echo "all done"

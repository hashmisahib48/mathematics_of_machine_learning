#!/bin/bash
# Run from anywhere; regenerates every verify/*.out of Chapter 33.
cd "$(dirname "$0")/.."
for n in toy scratch wrappers targets native ecoc ecoc_hand chain case_digits case_multilabel case_linnerud checks; do
  python3 -W error::FutureWarning -W error::DeprecationWarning code/$n.py > verify/$n.out || echo FAIL $n
done

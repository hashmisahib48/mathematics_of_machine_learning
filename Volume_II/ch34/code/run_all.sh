#!/bin/bash
# Run from anywhere; regenerates every verify/*.out of Chapter 34.
cd "$(dirname "$0")/.."
for n in toy sklearn_demo scratch_test pitfalls case_digits checks; do
  python3 -W error::FutureWarning -W error::DeprecationWarning code/$n.py > verify/$n.out || echo FAIL $n
done

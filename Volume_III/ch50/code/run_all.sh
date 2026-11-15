#!/bin/sh
# Reruns every script of Chapter 50; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
python3 $W code/cars.py || exit 1
for f in hand target_scratch sklearn_toy case_cars errors leak prose_numbers cardinality; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null

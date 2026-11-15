#!/bin/sh
# Reruns every script of Chapter 54; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in leak hand pipeline_scratch sklearn_toy case_cars errors prose_numbers sweeps; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 $W fig/eqs.py && python3 $W fig/figs.py > /dev/null && python3 $W fig/probs.py > /dev/null

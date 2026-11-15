#!/bin/sh
# Reruns every script of Chapter 51; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
cd code || exit 1
for f in hand knn_scratch sklearn_toy errors case_holes prose_numbers; do
  python3 $W $f.py > ../verify/$f.out || exit 1
done
cd .. || exit 1
python3 $W fig/eqs.py && python3 $W fig/figs.py > /dev/null && python3 $W fig/probs.py > /dev/null

#!/bin/sh
# Reruns every script of Chapter 38 (about 1 minute in all). Run from anywhere.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for s in toy toy_more dbscan_scratch dbscan_sklearn facts scaling case_gps; do
  (cd code && python3 $W $s.py > ../verify/$s.out) || echo "FAILED $s"
done
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null

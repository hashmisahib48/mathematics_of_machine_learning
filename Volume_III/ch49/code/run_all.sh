#!/bin/bash
# Reruns every script of Chapter 49 (run from anywhere).
cd "$(dirname "$0")/.."
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in story sk boxcox_scratch case_bc case_diabetes prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || echo "FAILED $f"
done
python3 fig/eqs.py; python3 $W fig/figs.py; python3 fig/probs.py > /dev/null

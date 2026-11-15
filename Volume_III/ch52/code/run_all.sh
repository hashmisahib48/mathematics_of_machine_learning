#!/bin/bash
# Reruns every script of Chapter 52 (run from anywhere).
cd "$(dirname "$0")/.."
W="-W error::FutureWarning -W error::DeprecationWarning"
cd code
for f in sk tfidf_scratch hashing case lsa errors prose_numbers; do
  python3 $W $f.py > ../verify/$f.out || echo "FAILED $f"
done
cd ..
python3 fig/eqs.py; python3 $W fig/figs.py; python3 fig/probs.py > /dev/null

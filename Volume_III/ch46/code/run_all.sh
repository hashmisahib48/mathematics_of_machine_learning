#!/bin/sh
# Reruns every script of Chapter 46; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in hand shrink_scratch sklearn_cov glasso_sklearn mcd_scratch mcd_wine case_returns others prose_numbers; do
  (cd code && python3 $W $f.py) > verify/$f.out 2>&1 || exit 1
done
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null
awk 'NR==1||$1==30||$1==480' verify/case_returns.out > verify/case_returns_short.out

#!/bin/sh
# Reruns every script of Chapter 48; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in toy sklearn_sensors lof_scratch thresholds errors ocsvm_nu sgd_nystroem timing case_cancer case_repeats prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null

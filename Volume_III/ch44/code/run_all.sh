#!/bin/sh
# Reruns every script of Chapter 44; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in toy_mds toy_lle toy_tsne mds_scratch smacof_scratch sklearn_manifold tsne_check digits_scan case_swiss case_digits prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 code/mds_warnings.py > verify/mds_warnings.out || exit 1
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/figs_digits.py && python3 $W fig/probs.py > /dev/null 2>&1

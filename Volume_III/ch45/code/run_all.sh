#!/bin/sh
# Reruns every script of Chapter 45; outputs go to verify/.
cd "$(dirname "$0")/.." || exit 1
W="-W error::FutureWarning -W error::DeprecationWarning"
for f in hand nystroem_scratch rp_sklearn case_spectra kernel_sklearn others case_digits timing prose_numbers; do
  python3 $W code/$f.py > verify/$f.out || exit 1
done
python3 $W fig/eqs.py && python3 $W fig/figs.py && python3 $W fig/probs.py > /dev/null

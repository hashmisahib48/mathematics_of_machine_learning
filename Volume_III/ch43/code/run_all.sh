#!/bin/sh
# Reruns every script of Chapter 43; outputs go to verify/ (run from any directory).
cd "$(dirname "$0")/.." || exit 1
FLAGS="-W error::FutureWarning -W error::DeprecationWarning"
for s in toy_nmf toy_ica nmf_scratch nmf_sklearn ica_sklearn fa_sklearn dict_sklearn lda_sklearn case_digits; do
  echo "== $s"; python3 $FLAGS code/$s.py > verify/$s.out || exit 1
done
python3 $FLAGS code/nmf_checks.py > verify/nmf_checks.out || exit 1
for f in eqs figs figs_b probs; do echo "== fig/$f"; python3 $FLAGS fig/$f.py || exit 1; done

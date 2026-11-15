#!/bin/bash
# Re-runs every script of Chapter 40: listings (outputs captured in ../verify), equations, figures and problems.
cd "$(dirname "$0")"
W="-W error::FutureWarning -W error::DeprecationWarning"
for n in toy gmm_test gmm_sklearn select_bic bayes_gmm case_wine case_digits prose_numbers errors param_count; do
  python3 $W $n.py > ../verify/$n.out 2>&1 || { echo "FAILED: $n"; exit 1; }
done
cd ..
for f in eqs figs1 figs2 figs3 figs4 figs5 probs; do
  python3 $W fig/$f.py > /dev/null || { echo "FAILED: fig/$f"; exit 1; }
done
echo "all scripts ok"

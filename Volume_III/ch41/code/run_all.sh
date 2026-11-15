#!/bin/bash
# Reruns every script of Chapter 41 (about 1.5 minutes); run from anywhere.
set -e
cd "$(dirname "$0")/.."
PY="python3 -W error::FutureWarning -W error::DeprecationWarning"
for n in toy scratch sklearn_toy case_wine stability errors prose_numbers; do
  if [ $n = errors ]; then python3 code/$n.py > verify/$n.out
  else $PY code/$n.py > verify/$n.out; fi
done
for f in eqs figs1 figs2 figs3 probs; do $PY fig/$f.py; done

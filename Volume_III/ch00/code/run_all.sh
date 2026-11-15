#!/bin/bash
# Reruns every script of the refresher (a few seconds); run from anywhere.
set -e
cd "$(dirname "$0")/.."
PY="python3 -W error::FutureWarning -W error::DeprecationWarning"
for n in api toolkit worked; do $PY code/$n.py > verify/$n.out; done
for f in eqs figs; do $PY fig/$f.py > /dev/null; done

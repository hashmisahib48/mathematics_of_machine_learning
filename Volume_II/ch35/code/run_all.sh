#!/bin/sh
# Re-run every listing and write its captured output to ../verify/<name>.out
cd "$(dirname "$0")" && mkdir -p ../verify
for f in spectra toy directions nipals_toy pls_test pls_sklearn case_spectra case_linnerud checks; do
  python3 -W error::FutureWarning -W error::DeprecationWarning $f.py > ../verify/$f.out 2>&1 || echo "FAILED $f"
done

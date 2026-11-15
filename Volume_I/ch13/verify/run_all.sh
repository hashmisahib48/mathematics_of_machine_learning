cd "$(dirname "$0")/../code"
for f in toy_huber toy_pairs huber_scratch robust_sklearn quantile_interval case_diabetes; do
  python3 -W error::FutureWarning -W error::DeprecationWarning $f.py > ../verify/$f.out 2>../verify/$f.err || echo FAIL $f
  [ -s ../verify/$f.err ] && echo "stderr in $f" && cat ../verify/$f.err
done

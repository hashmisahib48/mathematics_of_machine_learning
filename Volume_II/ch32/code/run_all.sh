cd "$(dirname "$0")"
for f in reliability dose_toy forecast_toy scores_toy iso_scratch calib_scratch iso_sklearn calib_sklearn case_cancer prose_numbers case_repeat; do
  python3 -W error::FutureWarning -W error::DeprecationWarning $f.py > ../verify/$f.out || echo FAIL $f
done

# Problem 32B.2: uniform and quantile bins of calibration_curve
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=4000, n_features=10,
                           n_informative=3, weights=[0.9, 0.1],
                           flip_y=0.02, random_state=0)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=0,
                                  stratify=y)
p = LogisticRegression().fit(Xa, ya).predict_proba(Xb)[:, 1]
print(f"test cases {len(yb)}, share of events {yb.mean():.4f}, "
      f"largest forecast {p.max():.3f}")
for strat in ("uniform", "quantile"):
    obs, pred = calibration_curve(yb, p, n_bins=5, strategy=strat)
    edges = (np.linspace(0, 1, 6) if strat == "uniform"
             else np.percentile(p, np.linspace(0, 100, 6)))
    k = np.searchsorted(edges[1:-1], p)           # bin of every case
    n = np.bincount(k, minlength=5)
    ne = np.bincount(k, weights=yb, minlength=5)  # events per bin
    print(f"\n{strat}: {len(obs)} bins returned")
    print(" cases per bin :", n)
    print(" events per bin:", ne.astype(int))
    print(" predicted     :", pred.round(3))
    print(" observed      :", obs.round(3))
    se = np.sqrt(obs * (1 - obs) / n[n > 0])
    print(" std. error    :", se.round(3))

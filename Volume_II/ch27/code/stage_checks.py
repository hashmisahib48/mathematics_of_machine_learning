# Checks of the loss bookkeeping, early stopping and importances.
import numpy as np
from sklearn.datasets import (load_diabetes, load_breast_cancer,
                              make_friedman1)
from sklearn.ensemble import (GradientBoostingRegressor as GR,
                              GradientBoostingClassifier as GC)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

X, y = load_diabetes(return_X_y=True)
g = GR(n_estimators=20, max_depth=2, random_state=0).fit(X, y)
mse = [mean_squared_error(y, p) for p in g.staged_predict(X)]
print("train_score_ = mean squared error:",
      np.allclose(g.train_score_, mse))

# Out-of-bag bookkeeping.
g = GR(n_estimators=30, subsample=0.5, random_state=0).fit(X, y)
d = g.oob_improvement_[1:] - (g.oob_scores_[:-1] - g.oob_scores_[1:])
print("oob_improvement_[m] = oob_scores_[m-1] - oob_scores_[m]:",
      np.abs(d).max() < 1e-9, " oob_score_ is oob_scores_[-1]:",
      g.oob_score_ == g.oob_scores_[-1])


# Early stopping: the rule of the source, replayed.
def replay(Xv, yv, g, n, tol, loss):
    hist = np.full(n, np.inf)
    for i, F in enumerate(g._staged_raw_predict(Xv)):
        v = loss(yv, F)
        if np.any(v + tol < hist):
            hist[i % n] = v
        else:
            return i + 1
    return g.n_estimators


for n_no in (1, 5, 20):
    g = GR(n_estimators=1000, learning_rate=0.1, n_iter_no_change=n_no,
           random_state=0).fit(X, y)
    Xt, Xv, yt, yv = train_test_split(X, y, random_state=0,
                                      test_size=0.1)
    ref = GR(n_estimators=g.n_estimators_, learning_rate=0.1,
             random_state=0).fit(Xt, yt)
    k = replay(Xv, yv, ref, n_no, 1e-4,
               lambda yy, F: np.mean((yy - F.ravel()) ** 2))
    print(f"n_iter_no_change={n_no:2d}: n_estimators_ = "
          f"{g.n_estimators_}, replayed rule: {k}")

# Importances: mean of the unnormalised tree importances.
g = GR(n_estimators=30, random_state=0).fit(X, y)
raw = np.mean([e[0].tree_.compute_feature_importances(normalize=False)
               for e in g.estimators_], axis=0)
print("feature_importances_ = normalised mean of tree importances:",
      np.allclose(g.feature_importances_, raw / raw.sum()))

# Stumps give an additive model; deeper trees do not.
Xf, yf = make_friedman1(n_samples=400, noise=0.5, random_state=0)
rng = np.random.RandomState(0)
A, B = rng.uniform(size=(2, 200, 10))
for d in (1, 2, 3):
    m = GR(max_depth=d, n_estimators=100, random_state=0).fit(Xf, yf)
    Ab = A.copy(); Ab[:, 0] = B[:, 0]            # change x1 only
    Ac = A.copy(); Ac[:, 1] = B[:, 1]            # change x2 only
    Abc = A.copy(); Abc[:, 0] = B[:, 0]; Abc[:, 1] = B[:, 1]
    inter = m.predict(Abc) - m.predict(Ab) - m.predict(Ac) + m.predict(A)
    print(f"max_depth={d}: largest |interaction of x1 and x2| "
          f"{np.abs(inter).max():.4f}")

# Subsampling and max_features arithmetic.
Xc, yc = load_breast_cancer(return_X_y=True)
g = GC(n_estimators=2, max_features=0.1, random_state=0).fit(Xc, yc)
print("max_features=0.1 of 30 features ->", g.max_features_,
      " 'sqrt' ->", GC(n_estimators=2, max_features="sqrt")
      .fit(Xc, yc).max_features_)

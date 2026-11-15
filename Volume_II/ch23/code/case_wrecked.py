# Diabetes with wrecked training targets: 15 % shifted by +-300.
import numpy as np
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.datasets import load_diabetes
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)


def scaled(est):                         # as in case_diabetes.py
    pipe = make_pipeline(StandardScaler(), est)
    return TTR(pipe, transformer=StandardScaler())


models = {                           # settings tuned on clean data
    "OLS": LinearRegression(),
    "ridge": RidgeCV(alphas=np.logspace(-3, 1, 41)),
    "KRR rbf": scaled(KernelRidge(alpha=1, kernel="rbf",
                                  gamma=0.0316)),
    "SVR linear": scaled(SVR(kernel="linear", C=0.0316,
                             epsilon=0.4)),
    "SVR rbf": scaled(SVR(C=1, epsilon=0.3, gamma=0.0316)),
}
n_bad = int(0.15 * len(y_tr))
scores = {k: [] for k in models}
for seed in range(5):
    rng = np.random.default_rng(seed)
    bad = rng.choice(len(y_tr), n_bad, replace=False)
    y_w = y_tr.copy()
    y_w[bad] += rng.choice([-300.0, 300.0], n_bad)
    for k, mdl in models.items():
        scores[k].append(mdl.fit(X_tr, y_w).score(X_te, y_te))
print("test R2 on clean patients; clean training, then 5 wrecked")
for k, mdl in models.items():
    clean = mdl.fit(X_tr, y_tr).score(X_te, y_te)
    s = np.array(scores[k])
    print(f"{k:11s} clean {clean:.3f}  wrecked mean {s.mean():.3f}"
          f"  min {s.min():.3f}  max {s.max():.3f}")

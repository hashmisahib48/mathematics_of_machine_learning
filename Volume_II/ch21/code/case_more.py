# Scaling and a second data set for kernel ridge regression.
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def model(scale, **kw):
    first = StandardScaler() if scale else "passthrough"
    pipe = make_pipeline(first, KernelRidge(kernel="rbf", **kw))
    ys = StandardScaler()
    return TransformedTargetRegressor(pipe, transformer=ys)


X, y = load_diabetes(return_X_y=True)
X[:, 2] *= 100                   # bmi in other units: spread 4.8
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)
for scale in (False, True):
    m = model(scale, gamma=0.03, alpha=1.0).fit(X_tr, y_tr)
    print(f"diabetes, bmi x 100, scaler={scale!s:5}: "
          f"test R2 {m.score(X_te, y_te):.3f}")

Xf, yf = make_friedman1(1300, noise=1.0, random_state=0)
X_tr, X_te, y_tr, y_te = Xf[:300], Xf[300:], yf[:300], yf[300:]
ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_tr, y_tr)
p = "regressor__kernelridge__"
grid = {p + "alpha": np.logspace(-4, 0, 9),
        p + "gamma": np.logspace(-3, 0, 7)}
gs = GridSearchCV(model(True), grid, cv=5).fit(X_tr, y_tr)
r2 = ridge.score(X_te, y_te)
print(f"friedman1, ridge        test R2 {r2:.3f}")
print(f"friedman1, RBF kernel   test R2 {gs.score(X_te, y_te):.3f}")
print("  chosen:", {k.split('__')[-1]: round(float(v), 3)
                    for k, v in gs.best_params_.items()})

# Case study: SVR on the diabetes data, split of Chapter 9.
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR, NuSVR

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)
raw = SVR().fit(X_tr, y_tr)          # defaults; y in its own units
print(f"default SVR, raw y    test R2 {raw.score(X_te, y_te):.3f}")


def model(est):                          # scale X; centre, scale y
    pipe = make_pipeline(StandardScaler(), est)
    ys = StandardScaler()
    return TransformedTargetRegressor(pipe, transformer=ys)


dflt = model(SVR()).fit(X_tr, y_tr)
print(f"default SVR, y scaled test R2 {dflt.score(X_te, y_te):.3f}")


def tune(name, est, **grid):
    step = type(est).__name__.lower()
    grid = {f"regressor__{step}__{k}": v for k, v in grid.items()}
    gs = GridSearchCV(model(est), grid, cv=5, scoring="r2")
    gs.fit(X_tr, y_tr)
    best = " ".join(f"{k.split('__')[-1]}={v:.3g}"
                    for k, v in gs.best_params_.items())
    sv = getattr(gs.best_estimator_.regressor_[-1], "support_", [])
    print(f"{name:10s} CV {gs.best_score_:.3f} test "
          f"{gs.score(X_te, y_te):.3f} SV {len(sv):3d} {best}")


C, G, L = np.logspace(-1, 1, 5), np.logspace(-3.5, -0.5, 7), \
    np.logspace(-3, 0, 7)
tune("SVR linear", SVR(kernel="linear"), C=L, epsilon=[.1, .4, .8])
tune("SVR rbf", SVR(), C=C, gamma=G, epsilon=[.1, .3, .6])
tune("NuSVR rbf", NuSVR(), C=C, gamma=G, nu=[.3, .5, .8])
tune("LinearSVR", LinearSVR(max_iter=20000), C=L)

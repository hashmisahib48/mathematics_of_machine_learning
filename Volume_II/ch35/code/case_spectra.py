# Case study: fat from 100-channel spectra, 60 training samples.
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectra import make_spectra

X, y = make_spectra(60, seed=0)         # training set
Xt, yt = make_spectra(1000, seed=1)     # new samples
cv = KFold(5, shuffle=True, random_state=0)
grid = {"n_components": range(1, 16)}
sc = lambda m: make_pipeline(StandardScaler(), m)
models = {
    "mean": DummyRegressor(),
    "OLS": sc(LinearRegression()),
    "RidgeCV": sc(RidgeCV(alphas=np.logspace(-3, 4, 30))),
    "PCA + OLS": GridSearchCV(
        make_pipeline(StandardScaler(), PCA(), LinearRegression()),
        {"pca__n_components": range(1, 16)}, cv=cv,
        scoring="neg_root_mean_squared_error"),
    "PLSRegression": GridSearchCV(
        PLSRegression(), grid, cv=cv,
        scoring="neg_root_mean_squared_error")}
print(f"{'model':15s}{'chosen':>8s}{'cv RMSE':>9s}"
      f"{'train':>8s}{'test':>8s}")
for name, m in models.items():
    m.fit(X, y)
    best = getattr(m, "best_params_", None)
    k = "" if best is None else list(best.values())[0]
    cvr = "" if best is None else f"{-m.best_score_:.2f}"
    tr = rmse(y, m.predict(X).ravel())
    te = rmse(yt, m.predict(Xt).ravel())
    print(f"{name:15s}{k:>8}{cvr:>9s}{tr:8.2f}{te:8.2f}")

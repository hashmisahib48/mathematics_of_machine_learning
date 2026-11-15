# Case study: kernel ridge on the diabetes data, split of Chapter 9.
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)
ols = LinearRegression().fit(X_tr, y_tr)
rcv = RidgeCV(alphas=np.logspace(-3, 1, 41)).fit(X_tr, y_tr)
print(f"OLS    test R2 {ols.score(X_te, y_te):.3f}")
print(f"ridge  test R2 {rcv.score(X_te, y_te):.3f}")


def model(kernel):                # scale X; centre and scale y
    krr = KernelRidge(kernel=kernel)
    pipe = make_pipeline(StandardScaler(), krr)
    ys = StandardScaler()
    return TransformedTargetRegressor(pipe, transformer=ys)


p = "regressor__kernelridge__"
alphas = np.logspace(-2, 2, 9)
gammas = np.logspace(-3.5, -0.5, 7)
grids = {
    "linear": {p + "alpha": np.logspace(-1, 4, 11)},
    "poly": {p + "alpha": alphas, p + "degree": [2, 3],
             p + "gamma": [0.01, 0.03, 0.1]},
    "rbf": {p + "alpha": alphas, p + "gamma": gammas},
    "laplacian": {p + "alpha": alphas, p + "gamma": gammas},
}
for kernel, grid in grids.items():
    gs = GridSearchCV(model(kernel), grid, cv=5, scoring="r2")
    gs.fit(X_tr, y_tr)
    best = {k.split("__")[-1]: round(float(v), 3)
            for k, v in gs.best_params_.items()}
    print(f"{kernel:9s} CV R2 {gs.best_score_:.3f}  "
          f"test R2 {gs.score(X_te, y_te):.3f}  {best}")

# Case study: which of the ten measurements does the lasso keep?
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import (ElasticNetCV, Lasso, LassoCV,
                                  LinearRegression)
from sklearn.model_selection import train_test_split

data = load_diabetes()
X, y = data.data, data.target
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0)       # split of Chapter 9

ols = LinearRegression().fit(X_tr, y_tr)
lasso = LassoCV(cv=5, random_state=0).fit(X_tr, y_tr)
print("alpha_ =", round(lasso.alpha_, 4), "  alpha_max =",
      round(lasso.alphas_[0], 4))
mean = lasso.mse_path_.mean(axis=1)             # one-s.e. rule
sem = lasso.mse_path_.std(axis=1, ddof=1) / np.sqrt(5)
best = mean.argmin()
a_1se = lasso.alphas_[mean <= mean[best] + sem[best]][0]
sparse = Lasso(alpha=a_1se).fit(X_tr, y_tr)
print("one-s.e. alpha =", round(a_1se, 4))

short = ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0)
short.fit(X_tr, y_tr)                           # default eps=1e-3
print("default path: alpha_ =", round(short.alpha_, 4),
      " last grid point:", bool(short.alpha_ == short.alphas_[-1]))
enet = ElasticNetCV(l1_ratio=0.5, eps=1e-4, cv=5, random_state=0)
enet.fit(X_tr, y_tr)                            # longer path
print("eps=1e-4: alpha_ =", f"{enet.alpha_:.5f}")

keep = np.flatnonzero(lasso.coef_)              # relaxed lasso
refit = LinearRegression().fit(X_tr[:, keep], y_tr)
print("\nmodel                test R2   features kept")
rows = [("OLS", ols.score(X_te, y_te), 10),
        ("LassoCV", lasso.score(X_te, y_te), len(keep)),
        ("LassoCV, 1 s.e.", sparse.score(X_te, y_te),
         np.sum(sparse.coef_ != 0)),
        ("ElasticNetCV, 0.5", enet.score(X_te, y_te),
         np.sum(enet.coef_ != 0)),
        ("lasso + OLS refit", refit.score(X_te[:, keep], y_te),
         len(keep))]
for name, r2, k in rows:
    print(f"{name:19s} {r2:8.3f}   {k}")

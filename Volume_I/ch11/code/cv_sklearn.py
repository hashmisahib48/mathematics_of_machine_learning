# Choosing alpha by cross-validation: LassoCV and ElasticNetCV.
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV

X, y, w_true = make_regression(
    n_samples=120, n_features=20, n_informative=4, noise=25.0,
    coef=True, random_state=2)
print("true non-zero features:", np.flatnonzero(w_true))

cv = LassoCV(alphas=100, cv=5, random_state=0).fit(X, y)
print("alpha_      =", round(cv.alpha_, 4))
print("alphas_     :", cv.alphas_.shape, "from",
      round(cv.alphas_[0], 2), "to", round(cv.alphas_[-1], 4))
print("mse_path_   :", cv.mse_path_.shape, "(alphas, folds)")
print("selected    :", np.flatnonzero(cv.coef_))

mean = cv.mse_path_.mean(axis=1)                # mean over folds
sem = cv.mse_path_.std(axis=1, ddof=1) / np.sqrt(5)
k = mean.argmin()
ok = mean <= mean[k] + sem[k]                   # one-s.e. rule
a1 = cv.alphas_[ok][0]                      # largest alpha
print("alpha, 1 s.e. rule =", round(a1, 4))
sparse = Lasso(alpha=a1).fit(X, y)
print("selected    :", np.flatnonzero(sparse.coef_))

en = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.9, 1.0], cv=5,
                  random_state=0).fit(X, y)
print("ElasticNetCV: l1_ratio_ =", en.l1_ratio_,
      " alpha_ =", round(en.alpha_, 4))
print("alphas_     :", en.alphas_.shape,
      " mse_path_ :", en.mse_path_.shape)
print("selected    :", np.flatnonzero(en.coef_))

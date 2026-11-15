import numpy as np
from scipy.stats import multivariate_normal as mvn
from sklearn.datasets import make_regression
from sklearn.linear_model import BayesianRidge, Ridge
from bayes_scratch import MyBayesRidge

X, y = make_regression(n_samples=60, n_features=5, n_informative=3,
                       noise=8.0, random_state=0)
mine = MyBayesRidge().fit(X, y)
ref = BayesianRidge().fit(X, y)
print("columns: scratch, scikit-learn")
print(f"alpha_   {mine.alpha_:.6g}  {ref.alpha_:.6g}")
print(f"lambda_  {mine.lambda_:.6g}  {ref.lambda_:.6g}")
for a, b in [(mine.coef_, ref.coef_), (mine.sigma_, ref.sigma_)]:
    assert np.allclose(a, b, rtol=1e-4, atol=1e-6)
assert np.isclose(mine.intercept_, ref.intercept_)
Xn = X[:5] + 0.3
m1, s1 = mine.predict(Xn, return_std=True)
m2, s2 = ref.predict(Xn, return_std=True)
assert np.allclose(m1, m2, rtol=1e-4)
assert np.allclose(s1, s2, rtol=1e-4)
print("std at 5 points:", np.round(s1, 3))

# fixed precisions: posterior mean equals ridge with lambda/alpha
fx = MyBayesRidge(fixed=(0.01, 0.5)).fit(X, y)
rd = Ridge(alpha=0.5 / 0.01).fit(X, y)
assert np.allclose(fx.coef_, rd.coef_)
print("fixed (0.01, 0.5) -> ridge alpha =", 0.5 / 0.01, ": equal")

# log evidence, Eq. 12.4-5 against Eq. 12.4-4 (hyperpriors off)
zero = dict(alpha_1=0, alpha_2=0, lambda_1=0, lambda_2=0)
br = BayesianRidge(compute_score=True, **zero).fit(X, y)
Xc, yc = X - X.mean(0), y - y.mean()
C = np.eye(60) / br.alpha_ + Xc @ Xc.T / br.lambda_   # Eq. 12.4-4
direct = mvn(np.zeros(60), C).logpdf(yc)
print("scores_ (first, last):", np.round(br.scores_[[0, -1]], 3))
print("normal density of y  :", round(direct, 3))
assert abs(direct - br.scores_[-1]) < 1e-2
up = bool(np.all(np.diff(br.scores_) >= 0))
print("never decreases:", up)

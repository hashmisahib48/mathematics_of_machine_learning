import numpy as np
from sklearn.linear_model import BayesianRidge

X = np.array([[-3.0], [-1.0], [1.0], [3.0]])   # one feature
y = np.array([-2.0, -2.0, 1.0, 3.0])

one = BayesianRidge(alpha_init=1.0, lambda_init=5.0, max_iter=1)
one.fit(X, y)                                  # Example 12.5-3
print(f"one step : alpha_ = {one.alpha_:.3f}  "
      f"lambda_ = {one.lambda_:.3f}")

br = BayesianRidge(compute_score=True).fit(X, y)
print(f"converged: alpha_ = {br.alpha_:.3f}  "
      f"lambda_ = {br.lambda_:.3f}  n_iter_ = {br.n_iter_}")
print("coef_      =", np.round(br.coef_, 4))
print("intercept_ =", br.intercept_)
print("sigma_     =", np.round(br.sigma_, 4))
print("scores_    =", np.round(br.scores_, 3))

Xnew = np.array([[0.0], [2.0], [6.0]])
mean, sd = br.predict(Xnew, return_std=True)
print("x*   =", Xnew.ravel())
print("mean =", np.round(mean, 3))
print("sd   =", np.round(sd, 3))

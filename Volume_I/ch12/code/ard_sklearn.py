import numpy as np
from sklearn.linear_model import ARDRegression, BayesianRidge

rng = np.random.default_rng(0)
n, p, hot = 100, 30, [2, 5, 11, 17]
w_true = np.zeros(p)
w_true[hot] = [3.0, -2.0, 1.5, 1.0]            # 4 of 30 matter
X = rng.normal(size=(n + 1000, p))
y = X @ w_true + rng.normal(0, 1.0, n + 1000)  # noise sd 1
Xtr, ytr, Xte, yte = X[:n], y[:n], X[n:], y[n:]

br = BayesianRidge().fit(Xtr, ytr)
for th in (1e4, 1e2):
    ard = ARDRegression(threshold_lambda=th).fit(Xtr, ytr)
    kept = np.flatnonzero(ard.lambda_ < th)
    print(f"threshold_lambda = {th:g}   n_iter_ = {ard.n_iter_}")
    print("  kept:", len(kept), "of", p, "  exact zeros:",
          int(np.sum(ard.coef_ == 0)))
    print("  weights of the 4:", np.round(ard.coef_[hot], 3))
    print("  lambda_ of the 4:", np.round(ard.lambda_[hot], 2))
    rest = np.delete(ard.lambda_, hot)
    print("  min lambda_ of the other 26:", round(rest.min(), 1))
    print("  noise sd:", round(1 / np.sqrt(ard.alpha_), 3),
          "  test R2:", round(ard.score(Xte, yte), 4))
print("true weights     :", w_true[hot])
print("BayesianRidge    :", np.round(br.coef_[hot], 3))
print("BayesianRidge  noise sd:", round(1 / np.sqrt(br.alpha_), 3),
      "  test R2:", round(br.score(Xte, yte), 4))

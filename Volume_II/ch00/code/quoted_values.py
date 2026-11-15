# Every number quoted in the text of the refresher that is not printed
# by one of the four listings. Writes verify/quoted_values.json.
import json
import numpy as np

n = {}
X = np.c_[np.ones(5), np.arange(1., 6)]
y = np.array([3., 4, 6, 7, 10])
H = 2 * X.T @ X
ev = np.linalg.eigvalsh(H)
n["hessian_eig"] = ev.round(4).tolist()
n["hessian_cond"] = float((ev[1] / ev[0]).round(4))
n["eta_max"] = float((2 / ev[1]).round(5))
w = np.linalg.solve(X.T @ X, X.T @ y)
r = y - X @ w
n["w"] = w.round(4).tolist()
n["sse"] = float((r @ r).round(4))
n["train_mse"] = float((r @ r / 5).round(4))
hat = X @ np.linalg.inv(X.T @ X) @ X.T
e = r / (1 - np.diag(hat))
n["loo_mse"] = float((e ** 2).mean().round(4))
n["loo_over_train"] = float(((e ** 2).mean() / (r @ r / 5)).round(3))
n["loo_squares"] = (e ** 2).round(4).tolist()
n["h_ii"] = np.diag(hat).round(4).tolist()
n["r"] = r.round(4).tolist()
n["e"] = (e + 0.0).round(4).tolist()
f = np.array([2., 1., 0.])
ex = np.exp(f)
n["exp_scores"] = ex.round(4).tolist()
n["exp_sum"] = float(ex.sum().round(4))
q = lambda l: 25 * l / (1 + l) - l
n["q"] = {str(l): round(q(l), 4) for l in (0, 2, 4, 6)}
n["gap"] = round(25 - q(2), 4)
n["f_star"] = float(np.sum((np.array([.6, .8]) - [3, 4]) ** 2).round(4))
P = ex / ex.sum()
n["P"] = P.round(4).tolist()
n["precision_x_recall_f1"] = round(2 * (4 / 6) * 0.8 / (4 / 6 + 0.8), 4)
json.dump(n, open("verify/quoted_values.json", "w"), indent=1)
print(json.dumps(n))

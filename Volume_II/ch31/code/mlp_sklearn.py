import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.simplefilter("ignore")           # no convergence notes
X = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1],
              [-2, 1], [-1, 2], [1, -2], [2, -1]], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
lr = LogisticRegression().fit(X, y)
print("logistic regression:", lr.score(X, y))

net = MLPClassifier((4,), solver="lbfgs", max_iter=500,
                    random_state=0).fit(X, y)
print("accuracy", net.score(X, y), " loss_", round(net.loss_, 4),
      " n_iter_", net.n_iter_)
print("coefs_     ", [w.shape for w in net.coefs_])
print("intercepts_", [b.shape for b in net.intercepts_])
print("n_layers_", net.n_layers_,
      " out_activation_", net.out_activation_)
p = net.predict_proba(X[[0, 4]])
print("P(class 1), samples 1 and 5:",
      " ".join(f"{v:.5f}" for v in p[:, 1]))

print("fits with accuracy 1 among 10 seeds (lbfgs):")
for act in ("relu", "tanh"):
    row = []
    for h in (2, 3, 4):
        ok = sum(MLPClassifier((h,), activation=act, solver="lbfgs",
                               max_iter=500, random_state=s)
                 .fit(X, y).score(X, y) == 1.0 for s in range(10))
        row.append(f"{h} units: {ok:2d}")
    print(f"  {act:5s}", "   ".join(row))

# SGD with log loss and LogisticRegression share one objective.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
n, alpha = len(y), 1e-2

# Eq. 17.5-1 with L2 penalty   <->   C * sum(loss) + 0.5 |w|^2
ref = LogisticRegression(C=1 / (n * alpha), tol=1e-10,
                         max_iter=10000).fit(X, y)
print("C = 1/(n alpha) =", round(1 / (n * alpha), 4))
for sched, extra in (("optimal", {}),
                     ("invscaling", dict(eta0=0.01, power_t=0.5))):
    sgd = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=2000,
                        tol=None, random_state=0,
                        learning_rate=sched, **extra).fit(X, y)
    dw = np.abs(sgd.coef_ - ref.coef_).max()
    b, b_ref = sgd.intercept_[0], ref.intercept_[0]
    print(f"{sched:10s} max |w - w_ref| = {dw:.4f}   "
          f"b = {b:.4f} (reference {b_ref:.4f})")

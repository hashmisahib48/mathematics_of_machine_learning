# Case study 2: Gaussian process classification on two moons.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)

gpc = GaussianProcessClassifier(
    ConstantKernel(1.0) * RBF(1.0), random_state=0).fit(Xtr, ytr)
print("kernel_:", gpc.kernel_)
print("log marginal likelihood:",
      round(gpc.log_marginal_likelihood_value_, 3))
lin = LogisticRegression().fit(Xtr, ytr)
models = [("GP classifier", gpc), ("logistic regression", lin)]
for name, m in models:
    p = m.predict_proba(Xte)
    print(f"{name:20s} train acc {m.score(Xtr, ytr):.3f}  "
          f"test acc {m.score(Xte, yte):.3f}  "
          f"test log loss {log_loss(yte, p):.3f}")
far = np.array([[6.0, 6.0]])
print("P(class 1) at (6, 6): GP",
      np.round(gpc.predict_proba(far)[0, 1], 3), " logistic",
      np.round(lin.predict_proba(far)[0, 1], 3))

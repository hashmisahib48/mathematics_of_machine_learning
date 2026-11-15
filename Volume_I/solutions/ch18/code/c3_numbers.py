# Problem 18C.3: which estimators can be fitted with a class of 8 samples and p = 12?
import warnings
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
np.set_printoptions(precision=4, suppress=True)

p, K = 12, 3
rng = np.random.default_rng(1)
mu = rng.normal(size=(K, p)) * 0.8
def draw(nk, rng):
    X, y = [], []
    for k, n in enumerate(nk):
        A = rng.normal(size=(p, p)) * 0.4 + np.eye(p) * (0.6 + 0.3 * k)
        X.append(mu[k] + rng.normal(size=(n, p)) @ A.T); y += [k] * n
    return np.vstack(X), np.array(y)
Xtr, ytr = draw([30, 30, 8], rng); Xte, yte = draw([2000, 2000, 2000], rng)
print("training sizes", np.bincount(ytr), " p =", p)
for k in range(K):
    Xk = Xtr[ytr == k]; C = np.cov(Xk.T, bias=True)
    print(f"class {k}: n_k = {len(Xk):2d}  rank of its covariance = {np.linalg.matrix_rank(C)}  (at most n_k - 1 = {len(Xk) - 1}, p = {p})")
Sw = sum(np.mean(ytr == k) * np.cov(Xtr[ytr == k].T, bias=True) for k in range(K))
print("pooled covariance: rank", np.linalg.matrix_rank(Sw), " (n - K =", len(ytr) - K, ")  smallest eigenvalue", np.linalg.eigvalsh(Sw)[0].round(4))

models = [("LDA, solver svd", LDA()), ("LDA, lsqr, shrinkage auto", LDA(solver="lsqr", shrinkage="auto")),
          ("QDA, plain", QDA()), ("QDA, reg_param=0.1", QDA(reg_param=0.1)), ("QDA, reg_param=1.0", QDA(reg_param=1.0)),
          ("QDA, eigen, shrinkage auto", QDA(solver="eigen", shrinkage="auto"))]
for name, m in models:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m.fit(Xtr, ytr)
        print(f"{name:28s} fits   test accuracy {m.score(Xte, yte):.3f}   train accuracy {m.score(Xtr, ytr):.3f}")
    except Exception as e:
        print(f"{name:28s} fails  {type(e).__name__}: {str(e)}")

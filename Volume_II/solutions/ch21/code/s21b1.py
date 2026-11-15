# Problem 21B.1: the sigmoid kernel added to MyKRR (Listing 21.1)
import sys, warnings
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import sigmoid_kernel
sys.path.insert(0, "../../../ch21/code")
from krr_scratch import MyKRR

class MyKRRSigmoid(MyKRR):
    def _k(self, A, B):                       # one more kernel
        if self.kernel == "sigmoid":
            g = self.gamma
            if g is None:
                g = 1.0 / A.shape[1]
            return np.tanh(g * (A @ B.T) + self.coef0)
        return super()._k(A, B)

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 3))
y = np.sin(X[:, 0]) + X[:, 1] ** 2
X_new = rng.normal(size=(5, 3))
kw = dict(alpha=1.0, kernel="sigmoid", gamma=0.5, coef0=0.0)
ours = MyKRRSigmoid(**kw).fit(X, y)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ref = KernelRidge(**kw).fit(X, y)
diff = abs(ours.predict(X_new) - ref.predict(X_new)).max()
print("largest difference of the predictions: %.1e" % diff)
for w in caught:
    print("warning:", str(w.message).split(".")[0])

K = sigmoid_kernel(X, gamma=0.5, coef0=0.0)       # the Gram matrix
ev = np.linalg.eigvalsh(K)
print("eigenvalues of K: %d negative, smallest %.4f, largest %.4f"
      % ((ev < 0).sum(), ev.min(), ev.max()))
ev1 = np.linalg.eigvalsh(K + np.eye(40))          # alpha = 1
cond = np.linalg.cond(K + np.eye(40))
print("K + I: %d negative, smallest %.4f, condition %.1f"
      % ((ev1 < 0).sum(), ev1.min(), cond))
print("eigenvalues of K in (-1, 0):", ((ev < 0) & (ev > -1)).sum())

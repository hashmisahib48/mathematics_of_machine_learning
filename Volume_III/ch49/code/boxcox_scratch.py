# Box-Cox lambda by maximum likelihood, from scratch (Section 49.6).
import numpy as np
from scipy.optimize import brent
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PowerTransformer

def psi(x, lam):                          # Eq. 49.3-5
    return np.log(x) if lam == 0 else (x**lam - 1) / lam

def neg_loglik(lam, x):                   # minus Eq. 49.3-7
    g = np.exp(np.log(x).mean())          # geometric mean of x
    return len(x) / 2 * np.log(psi(x / g, lam).var())

class BoxCox:
    def fit(self, X):
        self.lambdas_ = np.array([brent(neg_loglik, args=(c,),
            brack=(-2.0, 2.0)) for c in X.T])
        Z = self._map(X)
        self.mean_, self.scale_ = Z.mean(0), Z.std(0)
        return self
    def _map(self, X):
        return np.column_stack([psi(c, l) for c, l in
                                zip(X.T, self.lambdas_)])
    def transform(self, X):               # standardised, as sklearn
        return (self._map(X) - self.mean_) / self.scale_

X = load_breast_cancer().data[:, [0, 3, 5, 9]]   # positive columns
ref = PowerTransformer(method="box-cox").fit(X)
mine = BoxCox().fit(X)
assert np.allclose(mine.lambdas_, ref.lambdas_, atol=1e-6)
assert np.allclose(mine.transform(X), ref.transform(X), atol=1e-6)
print("scratch lambdas:", mine.lambdas_.round(4))
print("sklearn lambdas:", ref.lambdas_.round(4))
print("columns agree to 1e-6")

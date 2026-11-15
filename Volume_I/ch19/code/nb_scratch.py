import numpy as np
from scipy.special import logsumexp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class MyNB:
    """Naive Bayes; kind is 'multinomial' or 'gaussian'."""
    def __init__(self, kind="multinomial", alpha=1.0, vs=1e-9):
        self.kind, self.alpha, self.vs = kind, alpha, vs

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_ = np.unique(y)
        Xk = [X[y == k] for k in self.classes_]
        n_k = np.array([len(a) for a in Xk])
        self.class_log_prior_ = np.log(n_k / len(X))   # Eq. 19.3-6
        if self.kind == "multinomial":
            N = np.array([a.sum(0) for a in Xk]) + self.alpha
            N_k = N.sum(1, keepdims=True)         # Eq. 19.4-5
            self.feature_log_prob_ = np.log(N) - np.log(N_k)
        else:                                          # Eq. 19.4-2
            self.theta_ = np.array([a.mean(0) for a in Xk])
            eps = self.vs * X.var(0).max()
            self.var_ = np.array([a.var(0) for a in Xk]) + eps
        return self

    def _joint(self, X):
        X, lp = np.asarray(X, float), self.class_log_prior_
        if self.kind == "multinomial":                 # Eq. 19.4-6
            return lp + X @ self.feature_log_prob_.T
        d = X[:, None, :] - self.theta_          # shape (n, K, p)
        v = self.var_
        ll = -0.5 * (np.log(2 * np.pi * v) + d**2 / v)
        return lp + ll.sum(axis=2)                     # Eq. 19.4-3

    def predict_log_proba(self, X):
        j = self._joint(X)
        return j - logsumexp(j, axis=1, keepdims=True)  # Eq. 19.3-5

    def predict(self, X):
        return self.classes_[self._joint(X).argmax(axis=1)]


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    pairs = [("multinomial", MultinomialNB(alpha=0.5)),
             ("gaussian", GaussianNB())]
    for kind, ref in pairs:
        mine = MyNB(kind, alpha=0.5).fit(Xtr, ytr)
        ref.fit(Xtr, ytr)
        a = mine.predict_log_proba(Xte)
        b = ref.predict_log_proba(Xte)
        same = (mine.predict(Xte) == ref.predict(Xte)).all()
        assert np.allclose(a, b) and same
        acc = (mine.predict(Xte) == yte).mean()
        print(f"{kind:12s} max log-proba difference "
              f"{np.abs(a - b).max():.1e}, test accuracy {acc:.4f}")

# Label propagation and spreading in closed form (Section 34.7).
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class GraphSSL:
    def __init__(self, method="propagation", gamma=20.0, alpha=0.2):
        self.method, self.gamma, self.alpha = method, gamma, alpha

    def fit(self, X, y):
        self.X_, lab = X, y != -1
        self.classes_ = c = np.unique(y[lab])
        Y = (y[:, None] == c[None, :]) * 1.0      # one-hot rows
        W = rbf_kernel(X, gamma=self.gamma)       # Eq. 34.3-1
        np.fill_diagonal(W, 0.0)                  # no self-loops
        d = W.sum(axis=1)                         # degrees
        if self.method == "propagation":
            T, U = W / d[:, None], ~lab           # Eq. 34.3-3
            A = np.eye(U.sum()) - T[np.ix_(U, U)]
            F = Y.copy()                          # clamped rows
            F[U] = np.linalg.solve(A, T[np.ix_(U, lab)] @ Y[lab])
        else:
            S = W / np.sqrt(np.outer(d, d))       # Eq. 34.3-9
            A = np.eye(len(y)) - self.alpha * S
            F = (1 - self.alpha) * np.linalg.solve(A, Y)  # 34.3-10
        s = F.sum(axis=1, keepdims=True)
        self.label_distributions_ = F / np.where(s == 0, 1, s)
        self.transduction_ = c[F.argmax(axis=1)]
        return self

    def predict_proba(self, X):
        K = rbf_kernel(X, self.X_, gamma=self.gamma)  # 34.3-11
        P = K @ self.label_distributions_
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

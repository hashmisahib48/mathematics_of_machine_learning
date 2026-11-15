import numpy as np


def first_weights(X, Y, tol=1e-6, max_iter=500):
    u, w_old = Y[:, 0], 100.0           # start: first column of Y
    for _ in range(max_iter):
        w = X.T @ u / (u @ u)           # Eq. 35.4-4
        w /= np.linalg.norm(w)
        t = X @ w
        c = Y.T @ t / (t @ t)           # Eq. 35.4-5
        u = Y @ c / (c @ c)             # Eq. 35.4-6
        if ((w - w_old) ** 2).sum() < tol or Y.shape[1] == 1:
            break
        w_old = w
    return w * np.sign(w[np.argmax(np.abs(w))])

class MyPLS:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.one_d_, Y = Y.ndim == 1, Y.reshape(len(Y), -1)
        self.xm_, self.xs_ = X.mean(0), X.std(0, ddof=1)
        self.ym_, self.ys_ = Y.mean(0), Y.std(0, ddof=1)
        Xk = (X - self.xm_) / self.xs_
        Yk = (Y - self.ym_) / self.ys_
        W, P, Q = [], [], []
        for _ in range(self.n_components):
            w = first_weights(Xk, Yk)
            t = Xk @ w                          # Eq. 35.3-3
            p, q = Xk.T @ t / (t @ t), Yk.T @ t / (t @ t)
            Xk = Xk - np.outer(t, p)            # Eq. 35.3-6
            Yk = Yk - np.outer(t, q)
            W.append(w), P.append(p), Q.append(q)
        W, P, Q = np.array(W).T, np.array(P).T, np.array(Q).T
        self.x_rotations_ = W @ np.linalg.inv(P.T @ W)  # Eq. 35.3-8
        B = self.x_rotations_ @ Q.T             # Eq. 35.3-9
        self.coef_ = (B * self.ys_).T / self.xs_     # Eq. 35.3-10
        self.intercept_ = self.ym_
        return self

    def predict(self, X):
        out = (X - self.xm_) @ self.coef_.T + self.intercept_
        return out.ravel() if self.one_d_ else out

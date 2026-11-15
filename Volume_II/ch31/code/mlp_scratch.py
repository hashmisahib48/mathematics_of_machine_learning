import numpy as np
def glorot(rng, fan_in, fan_out):             # as scikit-learn does
    a = np.sqrt(6 / (fan_in + fan_out))
    return (rng.uniform(-a, a, (fan_in, fan_out)),
            rng.uniform(-a, a, fan_out))

class MyMLP:
    def __init__(self, hidden=8, alpha=1e-4, eta=0.5, n_iter=300):
        self.hidden, self.alpha = hidden, alpha
        self.eta, self.n_iter = eta, n_iter

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1            # Eq. 31.3-2
        H = np.maximum(Z1, 0)
        Z2 = H @ self.W2 + self.b2
        E = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
        return Z1, H, E / E.sum(axis=1, keepdims=True)

    def loss_grad(self, X, Y):
        n, a = len(X), self.alpha
        Z1, H, P = self.forward(X)
        pen = (self.W1**2).sum() + (self.W2**2).sum()
        J = -np.log((P * Y).sum(axis=1)).mean() + a * pen / (2 * n)
        D2 = P - Y                            # Eq. 31.4-1
        D1 = (D2 @ self.W2.T) * (Z1 > 0)      # Eq. 31.4-3
        return J, [(X.T @ D1 + a * self.W1) / n, D1.mean(axis=0),
                   (H.T @ D2 + a * self.W2) / n, D2.mean(axis=0)]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_) * 1.0
        rng = np.random.RandomState(0)
        self.W1, self.b1 = glorot(rng, X.shape[1], self.hidden)
        self.W2, self.b2 = glorot(rng, self.hidden, Y.shape[1])
        par = (self.W1, self.b1, self.W2, self.b2)
        self.loss_curve_ = []
        for _ in range(self.n_iter):          # full-batch descent
            J, grads = self.loss_grad(X, Y)
            self.loss_curve_.append(J)
            for q, g in zip(par, grads):
                q -= self.eta * g             # Eq. 31.5-2
        return self

    def predict(self, X):
        return self.classes_[self.forward(X)[2].argmax(axis=1)]

# Problem 31B.2: tanh hidden units, from scratch and in scikit-learn.
import sys, warnings
sys.path.insert(0, '../../../ch31/code')
import numpy as np
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mlp_scratch import MyMLP            # Listing 31.3
warnings.simplefilter("ignore", ConvergenceWarning)
class TanhMLP(MyMLP):
    def forward(self, X):
        H = np.tanh(X @ self.W1 + self.b1)       # tanh, not ReLU
        Z2 = H @ self.W2 + self.b2
        E = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
        return None, H, E / E.sum(axis=1, keepdims=True)
    def loss_grad(self, X, Y):
        n, a = len(X), self.alpha
        _, H, P = self.forward(X)
        pen = (self.W1**2).sum() + (self.W2**2).sum()
        J = -np.log((P * Y).sum(axis=1)).mean() + a * pen / (2 * n)
        D2 = P - Y
        D1 = (D2 @ self.W2.T) * (1 - H**2)       # Eq. 31.3-4
        return J, [(X.T @ D1 + a * self.W1) / n, D1.mean(axis=0),
                   (H.T @ D2 + a * self.W2) / n, D2.mean(axis=0)]
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
mine = TanhMLP(alpha=0.1, eta=0.5, n_iter=100).fit(X, y)
ref = MLPClassifier((8,), activation="tanh", solver="sgd", momentum=0,
                    alpha=0.1, learning_rate_init=0.5, max_iter=100,
                    batch_size=len(X), tol=0,
                    random_state=0).fit(X, y)
relu = MyMLP(alpha=0.1, eta=0.5, n_iter=100).fit(X, y)
print(f"final loss: mine {mine.loss_curve_[-1]:.6f}"
      f"  scikit-learn {ref.loss_curve_[-1]:.6f}")
d = abs(mine.W1 - ref.coefs_[0]).max()
print(f"largest difference of W1: {d:.1e}")
print("same predictions:", (mine.predict(X) == ref.predict(X)).all())
print(f"ReLU class, loss at 100: {relu.loss_curve_[-1]:.6f}")

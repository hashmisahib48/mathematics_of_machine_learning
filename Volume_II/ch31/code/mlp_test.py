import warnings
import numpy as np
from sklearn.datasets import load_iris
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from mlp_scratch import MyMLP

warnings.simplefilter("ignore", ConvergenceWarning)
X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Y = (y[:, None] == np.unique(y)) * 1.0

net = MyMLP(alpha=0.1, n_iter=5).fit(X, y)    # a point to test at
par = (net.W1, net.b1, net.W2, net.b2)
worst = 0.0
for q, g in zip(par, net.loss_grad(X, Y)[1]):
    for k in np.ndindex(q.shape):         # Eq. 3.10-1 (Volume I)
        old = q[k]
        q[k] = old + 1e-6; Jp = net.loss_grad(X, Y)[0]
        q[k] = old - 1e-6; Jm = net.loss_grad(X, Y)[0]
        q[k] = old
        worst = max(worst, abs((Jp - Jm) / 2e-6 - g[k]))
print(f"largest gradient error {worst:.1e}")

mine = MyMLP(alpha=0.1, n_iter=300).fit(X, y)
ref = MLPClassifier((8,), solver="sgd", momentum=0, alpha=0.1,
                    learning_rate_init=0.5, batch_size=len(X),
                    max_iter=300, tol=0, random_state=0).fit(X, y)
print(f"final loss {mine.loss_curve_[-1]:.6f} {ref.loss_:.6f}")
dW = abs(mine.W1 - ref.coefs_[0]).max()
print(f"largest difference of W1: {dW:.1e}")
same = (mine.predict(X) == ref.predict(X)).all()
acc = (mine.predict(X) == y).mean()
print("same predictions:", same, " accuracy", acc)

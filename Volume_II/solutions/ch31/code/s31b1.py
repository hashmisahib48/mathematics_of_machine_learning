# Problem 31B.1: gradient check with and without the penalty term.
import sys
sys.path.insert(0, '../../../ch31/code')
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from mlp_scratch import MyMLP            # Listing 31.3

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
Y = (y[:, None] == np.unique(y)) * 1.0

def check(alpha, drop_penalty=False):
    net = MyMLP(alpha=alpha, n_iter=5).fit(X, y)   # point to test at
    par = (net.W1, net.b1, net.W2, net.b2)
    grads = net.loss_grad(X, Y)[1]
    if drop_penalty:                               # drop alpha W/n
        grads = [grads[0] - alpha * net.W1 / len(X), grads[1],
                 grads[2] - alpha * net.W2 / len(X), grads[3]]
    worst = 0.0
    for q, g in zip(par, grads):
        for k in np.ndindex(q.shape):              # Eq. 3.10-1
            old = q[k]
            q[k] = old + 1e-6; Jp = net.loss_grad(X, Y)[0]
            q[k] = old - 1e-6; Jm = net.loss_grad(X, Y)[0]
            q[k] = old
            worst = max(worst, abs((Jp - Jm) / 2e-6 - g[k]))
    return worst

print(f"alpha = 0                        : {check(0.0):.1e}")
print(f"alpha = 3                        : {check(3.0):.1e}")
print(f"alpha = 3, penalty term left out : {check(3.0, True):.1e}")
net = MyMLP(alpha=3.0, n_iter=5).fit(X, y)
W = np.concatenate([net.W1.ravel(), net.W2.ravel()])
print("largest entry of the left-out term 3 W/n:",
      f"{np.abs(3.0 * W / len(X)).max():.4f}")

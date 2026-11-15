# Problem 33C.2: chain of ridge regressors against independent ridge
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import ClassifierChain as CC
from sklearn.multioutput import MultiOutputClassifier as MOC
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.tree import DecisionTreeClassifier

X, Y = load_linnerud(return_X_y=True)
for a in (0.01, 1.0, 100.0):
    ch = RegressorChain(Ridge(alpha=a)).fit(X, Y)
    mo = MultiOutputRegressor(Ridge(alpha=a)).fit(X, Y)
    d = np.abs(ch.predict(X) - mo.predict(X)).max()
    print(f"alpha {a:6g}: largest prediction difference {d:.1e}")
# algebra: w + c u equals the ridge weights of y2 on the features
Xc, Yc = X - X.mean(0), Y - Y.mean(0)
A = Xc.T @ Xc + 1.0 * np.eye(3)
u = np.linalg.solve(A, Xc.T @ Yc[:, 0])               # link 1
Z = np.c_[Xc, Yc[:, 0]]                         # link 2 sees y1
B = Z.T @ Z + 1.0 * np.eye(4)
wc = np.linalg.solve(B, Z.T @ Yc[:, 1])
v = np.linalg.solve(A, Xc.T @ Yc[:, 1])        # ridge of y2 alone
print("w + c u =", (wc[:3] + wc[3] * u).round(6))
print("ridge y2 =", v.round(6), " max difference",
      f"{np.abs(wc[:3] + wc[3] * u - v).max():.1e}")

# the same test for logistic regression and trees
def make(n, rng):
    X = rng.uniform(-2, 2, (n, 2)); y1 = X[:, 0] > 0
    return X, np.c_[y1, y1 & (X[:, 1] > 0)].astype(int)
rng = np.random.default_rng(0)
Xtr, Ytr = make(300, rng); Xte, Yte = make(2000, rng)
tree = DecisionTreeClassifier(max_depth=2, random_state=0)
for name, base in (("logistic", LogisticRegression(C=1e4, max_iter=1000)),
                   ("tree depth 2", tree)):
    ch = CC(base, order=[0, 1]).fit(Xtr, Ytr).predict(Xte)
    mo = MOC(base).fit(Xtr, Ytr).predict(Xte)
    nd = (ch != mo).any(1).sum()
    print(f"{name:13s} predictions differ in {nd:4d} of 2000; subset "
          f"accuracy chain {(ch == Yte).all(1).mean():.4f}, "
          f"independent {(mo == Yte).all(1).mean():.4f}")

# Problem 17B.2: stopping rules of SGDClassifier (breast cancer).
import contextlib, io, re
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
print("early_stopping   tol      n_iter_  test accuracy")
for es in (False, True):
    for tol in (1e-2, 1e-3, 1e-4):
        m = SGDClassifier(tol=tol, early_stopping=es,
                          random_state=0).fit(Xtr, ytr)
        print(f"{str(es):14s}  {tol:7g}  {m.n_iter_:7d}  "
              f"{m.score(Xte, yte):.4f}")

nv = int(np.ceil(0.1 * len(ytr)))      # validation_fraction=0.1
print("validation patients:", nv, " one patient:", round(1 / nv, 4))
# the objective of each epoch, as scikit-learn prints it (verbose=1)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    SGDClassifier(random_state=0, verbose=1).fit(Xtr, ytr)
J = re.findall(r"Objective: ([0-9.]+)", buf.getvalue())
J = np.array(J, float)
print("objective, epochs 24-33:")
print(J[23:].round(3))
print("lowest at epoch", J.argmin() + 1, " largest rise",
      np.diff(J).max().round(3))
for tol in (1e-2, 1e-3, 1e-4):         # rule of Section 17.5
    best, cnt = np.inf, 0
    for k, v in enumerate(J, 1):
        cnt = cnt + 1 if v > best - tol else 0
        best = min(best, v)
        if cnt == 5: break
    print(f"rule with tol={tol:g} stops at epoch {k}")

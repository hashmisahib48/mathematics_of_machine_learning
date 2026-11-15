# Problem 22B.2: support vectors and street width against C.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)
Z, Zt = sc.transform(X_tr), sc.transform(X_te)
s = 2 * y_tr - 1                          # labels -1, +1
print(f"n = {len(y_tr)}")
print("      C  SV  atC  in  width  sum_a  train  test")
for C in (0.001, 0.01, 0.1, 1, 10, 100):
    m = SVC(kernel="linear", C=C).fit(Z, y_tr)
    a = np.abs(m.dual_coef_[0])           # alpha of the SVs
    yf = s * m.decision_function(Z)       # y f of every sample
    w = 2 / np.linalg.norm(m.coef_)       # width of the street
    print(f"{C:7g}{len(m.support_):4d}{(a > C - 1e-8).sum():5d}"
          f"{(yf < 1 - 1e-6).sum():4d}{w:7.3f}{a.sum():7.2f}"
          f"{m.score(Z, y_tr):7.3f}{m.score(Zt, y_te):6.3f}")

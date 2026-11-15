# Problem 22B.1: gamma="scale" against gamma="auto".
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)
data = {"raw": (X_tr, X_te),
        "std": (sc.transform(X_tr), sc.transform(X_te))}
p = X.shape[1]
print("data  gamma     value  test acc  SV  mean g*d2  classes")
for name, (A, B) in data.items():
    # squared distances of all pairs of training samples
    D = (A[:, None] - A[None]) ** 2
    d2 = D.sum(-1)[np.triu_indices(len(A), 1)]
    for mode in ("scale", "auto"):
        g = 1 / p                              # "auto"
        if mode == "scale":
            g = 1 / (p * A.var())              # Eq. 22.5-4
        m = SVC(gamma=mode).fit(A, y_tr)
        cls = np.unique(m.predict(B))
        acc, nsv = m.score(B, y_te), len(m.support_)
        print(f"{name:5s} {mode:6s}{g:9.2e}{acc:9.4f}{nsv:5d}"
              f"{g * d2.mean():11.2f}  {cls}")
v = np.sort(X_tr.var(0))[::-1]
print("two widest columns, share of the variances:",
      round(v[:2].sum() / v.sum(), 4))
print("benign share of the test set:", round(y_te.mean(), 4))

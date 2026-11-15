# Problem 51B.1: how many neighbours for KNNImputer?
import sys
import numpy as np
sys.path.append("../../../ch51/code")   # book helpers (run here)
from sklearn.impute import KNNImputer, SimpleImputer
from holes import split, punch, HOLE

Xtr, Xte, ytr, yte = split()             # 331 training patients
A = punch(Xtr, "MCAR", 0.2, 0)           # 20 % of 4 columns blank
M = np.isnan(A)                          # where the holes are
print("blank cells:", M.sum(), " per column:", M[:, HOLE].sum(0))

def error(F):                            # RMSE on the blanks of a
    e = []                               # column / its true sd,
    for j in HOLE:                       # averaged over the 4 columns
        d = F[M[:, j], j] - Xtr[M[:, j], j]
        e.append(np.sqrt(np.mean(d ** 2)) / Xtr[:, j].std())
    return np.mean(e)

print("mean fill      %.3f" % error(SimpleImputer().fit_transform(A)))
print("    k   uniform  distance")
for k in (1, 3, 10, 30):
    row = []
    for w in ("uniform", "distance"):
        imp = KNNImputer(n_neighbors=k, weights=w)
        row.append(error(imp.fit_transform(A)))
    print("%5d   %.3f    %.3f" % (k, *row))

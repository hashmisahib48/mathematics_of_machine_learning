# Case study: breast cancer with holes in the table (Section 28.10).
import os
os.environ["OMP_NUM_THREADS"] = "1"       # small data: one thread
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X0, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
holes = {"complete": np.zeros(X0.shape, bool),
         "15% at random": rng.random(X0.shape) < 0.15}
skip = np.zeros(X0.shape, bool)       # worst-* values skipped
for j in range(20, 30):               # more often when benign
    skip[:, j] = rng.random(len(y)) < np.where(y == 1, 0.5, 0.05)
holes["skipped if benign"] = skip

hgb = HistGradientBoostingClassifier(
    min_samples_leaf=10, l2_regularization=1, random_state=0)
lr = lambda ind: make_pipeline(
    SimpleImputer(strategy="median", add_indicator=ind),
    StandardScaler(), LogisticRegression(max_iter=1000))
models = {"HGB, NaN as is": hgb,
          "HGB, median fill": make_pipeline(
              SimpleImputer(strategy="median"), hgb),
          "LogReg, median": lr(False),
          "LogReg, median+flags": lr(True)}

print("5-fold accuracy / test accuracy; columns:")
for k, m in enumerate(models, 1):
    print(" ", k, m)
print()
for name, mask in holes.items():
    X = np.where(mask, np.nan, X0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    cells = []
    for m in models.values():
        cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
        te = m.fit(Xtr, ytr).score(Xte, yte)
        cells.append("%.3f/%.3f" % (cv, te))
    print("%-18s %s  (NaN %.2f)" % (name, "  ".join(cells),
                                     mask.mean()))

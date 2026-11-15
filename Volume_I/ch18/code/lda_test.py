# Test: MyLDA against scikit-learn on the wine data (Listing 18.3).
import numpy as np
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA)
from lda_scratch import MyLDA

X, y = load_wine(return_X_y=True)
mine = MyLDA().fit(X, y)
for solver in ("svd", "lsqr", "eigen"):
    ref = LDA(solver=solver).fit(X, y)
    dp = np.abs(mine.predict_proba(X) - ref.predict_proba(X)).max()
    same = np.array_equal(mine.predict(X), ref.predict(X))
    assert dp < 1e-9 and same
    print(f"{solver:5s} max |dP| = {dp:.1e}  same labels: {same}")
ref = LDA().fit(X, y)
Z, Zr = mine.transform(X), ref.transform(X)
assert np.allclose(abs(Z), abs(Zr))
print("transform equal up to sign:", True)
print("explained ratio:",
      np.round(mine.explained_variance_ratio_, 4),
      np.round(ref.explained_variance_ratio_, 4))
sh = LDA(solver="lsqr", shrinkage=0.3).fit(X, y)
p_mine = MyLDA(0.3).fit(X, y).predict_proba(X)
assert np.allclose(p_mine, sh.predict_proba(X))
print("shrinkage 0.3 agrees: True")

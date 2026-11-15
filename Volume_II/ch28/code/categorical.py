# A column of 12 sites: native, ordinal and one-hot coding.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.preprocessing import OneHotEncoder

rng = np.random.default_rng(0)
names = np.array(list("ABCDEFGHIJKL"))
site = rng.integers(0, 12, 3000)
good = np.isin(site, [1, 4, 5, 8, 10])    # sites B, E, F, I, K
y = (rng.random(3000) < np.where(good, 0.85, 0.15)).astype(int)
tr, te = slice(0, 2000), slice(2000, 3000)
frame = pd.DataFrame({"site": pd.Categorical(names[site],
                                             categories=names)})
codes = site.reshape(-1, 1).astype(float)
onehot = OneHotEncoder(sparse_output=False).fit_transform(frame)

# The stump's left set, and the order of Eq. 28.3-12.
stump = HGB(max_iter=1, max_leaf_nodes=2, learning_rate=1.0,
            early_stopping=False).fit(frame[tr], y[tr])
bits = stump._predictors[0][0].raw_left_cat_bitsets[0][0]
print("left set:", names[[c for c in range(12) if bits >> c & 1]])
p = y[tr].mean()
g, h = p - y[tr], p * (1 - p)
key = [g[site[tr] == c].sum() / (h * (site[tr] == c).sum() + 10)
       for c in range(12)]
print("order   :", names[np.argsort(key)])

print("test accuracy by number of leaves")
print("leaves  native  ordinal  one-hot")
for L in (2, 3, 6, 9, 31):
    acc = []
    for X, cat in ((frame, "from_dtype"), (codes, None),
                   (onehot, None)):
        m = HGB(max_iter=1, max_leaf_nodes=L, learning_rate=1.0,
                early_stopping=False, categorical_features=cat)
        acc.append(m.fit(X[tr], y[tr]).score(X[te], y[te]))
    print("%5d   %6.3f  %7.3f  %7.3f" % (L, *acc))

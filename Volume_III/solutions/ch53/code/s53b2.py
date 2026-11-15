# Problem 53B.2: RFE on raw and on standardised training columns
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

d = load_breast_cancer()
Xa, Xb, ya, yb = train_test_split(
    d.data, d.target, test_size=0.25,
    random_state=0, stratify=d.target)
Xs = StandardScaler().fit_transform(Xa)     # fitted on training rows
lr = lambda: LogisticRegression(max_iter=10000)

def chosen(X):
    return RFE(lr(), n_features_to_select=5).fit(X, ya)

raw, std = chosen(Xa), chosen(Xs)
sd = Xa.std(axis=0)                         # spread of raw columns
for name, r in (("raw", raw), ("scaled", std)):
    idx = np.flatnonzero(r.support_)
    print(f"{name:7s}", idx.tolist())
    print("   sd of these:", sd[idx].round(3).tolist())
print("in common:", int((raw.support_ & std.support_).sum()))
print("sd of 30 columns: min %.4f, median %.3f, max %.1f"
      % (sd.min(), np.median(sd), sd.max()))

w = lr().fit(Xa, ya).coef_[0]               # raw weights
print("col     sd   |w| raw  rank raw  rank scaled (1 = kept)")
for j in np.flatnonzero(sd > 10):           # the widest columns
    print(f"{j:3d} {sd[j]:6.1f} {abs(w[j]):9.4f} {raw.ranking_[j]:9d}"
          f" {std.ranking_[j]:11d}")

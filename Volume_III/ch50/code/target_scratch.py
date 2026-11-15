import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
from cars import make_cars

class MyTargetEncoder:
    def __init__(self, smooth="auto"):
        self.smooth = smooth
    def fit(self, x, y):
        self.cats_, self.mean_ = np.unique(x), y.mean()
        self.enc_ = np.empty(len(self.cats_))
        for k, c in enumerate(self.cats_):
            yc = y[x == c]; n = len(yc)
            m = yc.var() / y.var() if self.smooth == "auto" \
                else self.smooth                     # Eq. 50.4-3
            lam = n / (n + m)                        # Eq. 50.4-1
            self.enc_[k] = lam * yc.mean() + (1 - lam) * self.mean_
        return self
    def transform(self, x):
        cats = self.cats_
        i = np.searchsorted(cats, x).clip(max=len(cats) - 1)
        known = cats[i] == x
        return np.where(known, self.enc_[i], self.mean_)
    def fit_transform(self, x, y, folds):            # cross fitting
        out = np.empty(len(x))
        for tr, te in folds.split(x):
            f = MyTargetEncoder(self.smooth).fit(x[tr], y[tr])
            out[te] = f.transform(x[te])
        return out

df = make_cars(1500, seed=1)
x, y = df.city.to_numpy(), df.price.to_numpy()
new = np.append(x[:5], "C9999")                      # unseen city
kf = KFold(5, shuffle=True, random_state=0)
for sm in ("auto", 5.0):
    sk = TargetEncoder(smooth=sm, target_type="continuous", cv=kf)
    mine = MyTargetEncoder(sm).fit(x, y)
    t = sk.fit(x[:, None], y).transform(new[:, None])[:, 0]
    a = MyTargetEncoder(sm).fit_transform(x, y, kf)
    b = sk.fit_transform(x[:, None], y)[:, 0]
    d1, d2 = abs(mine.transform(new) - t).max(), abs(a - b).max()
    print(f"smooth={sm}: transform {d1:.1e}, fit_trans {d2:.1e}")
    assert d1 < 1e-9 and d2 < 1e-9
print("unseen city:", mine.transform(new)[-1].round(3),
      "= mean", y.mean().round(3))

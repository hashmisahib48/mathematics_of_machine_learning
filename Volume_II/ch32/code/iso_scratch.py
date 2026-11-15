# Isotonic regression from scratch: PAVA and interpolation.
import numpy as np
from sklearn.isotonic import IsotonicRegression

def pava(y, w):                       # sorted data, Eq. 32.3-4
    val, wt, size = [], [], []
    for yi, wi in zip(y, w):
        val.append(yi); wt.append(wi); size.append(1)
        while len(val) > 1 and val[-2] > val[-1]:
            v, u, c = val.pop(), wt.pop(), size.pop()
            val[-1] = (val[-1] * wt[-1] + v * u) / (wt[-1] + u)
            wt[-1] += u
            size[-1] += c
    return np.repeat(val, size)

class MyIsotonic:
    def fit(self, x, y, w):
        on = w > 0                           # zero weights dropped
        ux, k = np.unique(x[on], return_inverse=True)
        wu = np.bincount(k, weights=w[on])   # ties pooled first
        yu = np.bincount(k, weights=w[on] * y[on]) / wu
        f = pava(yu, wu)
        keep = np.ones(len(ux), bool)        # ends of blocks only
        keep[1:-1] = (f[1:-1] != f[:-2]) | (f[1:-1] != f[2:])
        self.x_, self.y_ = ux[keep], f[keep]
        return self

    def predict(self, x):                    # Eq. 32.3-5
        x = np.clip(x, self.x_[0], self.x_[-1])
        return np.interp(x, self.x_, self.y_)


rng = np.random.default_rng(0)
x = np.round(rng.uniform(0, 10, 300), 1)     # many ties
y = np.sqrt(x) + rng.normal(0, 1, 300)
w = rng.integers(0, 4, 300).astype(float)    # some zeros
mine = MyIsotonic().fit(x, y, w)
ref = IsotonicRegression(out_of_bounds="clip")
ref.fit(x, y, sample_weight=w)
g = np.linspace(-1, 11, 500)
print("thresholds:", len(mine.x_), len(ref.X_thresholds_))
d = np.abs(mine.predict(g) - ref.predict(g)).max()
print("max |difference|:", d)
assert d < 1e-9

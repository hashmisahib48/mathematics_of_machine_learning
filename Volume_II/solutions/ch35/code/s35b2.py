# Problem 35B.2: the scale of a feature (Linnerud data).
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud

d = load_linnerud()
X, Y = d.data.copy(), d.target       # Chins, Situps, Jumps
print("sd of the columns:", X.std(0, ddof=1).round(2))
X100 = X.copy()
X100[:, 1] *= 100                    # Situps counted in hundredths
print("                            |w1|: Chins, Situps, Jumps")
for name, data, sc in (("raw, scale=False", X, False),
                       ("raw, scale=True", X, True),
                       ("Situps x 100, scale=False", X100, False),
                       ("Situps x 100, scale=True", X100, True)):
    pls = PLSRegression(1, scale=sc).fit(data, Y)
    print(f"{name:27s}", np.abs(pls.x_weights_[:, 0]).round(4))

# the same weights by hand, Eq. 35.4-2: first left singular vector
# of X^T Y (columns centred; standardised when scale=True)
def first(Xm, Ym, scale):
    A, B = Xm - Xm.mean(0), Ym - Ym.mean(0)
    if scale:
        A, B = A / A.std(0, ddof=1), B / B.std(0, ddof=1)
    return np.abs(np.linalg.svd(A.T @ B)[0][:, 0]).round(4)
print("by hand, Situps x 100, unscaled:    ", first(X100, Y, False))
print("by hand, Situps x 100, standardised:", first(X100, Y, True))

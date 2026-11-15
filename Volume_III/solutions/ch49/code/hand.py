# Hand checks for Problems 49A.1 and 49A.2 (NumPy against scikit-learn / SciPy)
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ---- 49A.1
x = np.array([2.0, 4, 6, 8, 30]); new = np.array([12.0, 40])
mean = x.mean(); dev2 = (x - mean) ** 2; var = dev2.mean(); sd = np.sqrt(var)
print("49A.1  mean", mean, " squared deviations", dev2, " sum", dev2.sum(), " var", var, " sd %.4f" % sd)
print("standard :", ((new - mean) / sd).round(4))
lo, hi = x.min(), x.max()
print("min, max, range", lo, hi, hi - lo, " minmax :", ((new - lo) / (hi - lo)).round(4), " clipped:", np.clip((new - lo) / (hi - lo), 0, 1).round(4))
q1, med, q3 = np.percentile(x, [25, 50, 75])
print("quartiles", q1, med, q3, " IQR", q3 - q1, " robust :", ((new - med) / (q3 - q1)).round(4))
tr = x.reshape(-1, 1); nw = new.reshape(-1, 1)
assert np.allclose(StandardScaler().fit(tr).transform(nw).ravel(), (new - mean) / sd)
assert np.allclose(MinMaxScaler().fit(tr).transform(nw).ravel(), (new - lo) / (hi - lo))
assert np.allclose(MinMaxScaler(clip=True).fit(tr).transform(nw).ravel(), [(12 - 2) / 28, 1.0])
assert np.allclose(RobustScaler().fit(tr).transform(nw).ravel(), (new - med) / (q3 - q1))
print("scikit-learn agrees")
# what 30 does: the same scalers fitted without the outlier
y = x[:4]; print("without the 30: sd %.4f, range %g, IQR %g" % (y.std(), np.ptp(y), np.subtract(*np.percentile(y, [75, 25]))))
print("40 in units of the training spread: standard %.3f, minmax %.3f, robust %.3f" % ((40 - mean) / sd, (40 - lo) / (hi - lo), (40 - med) / (q3 - q1)))

s1 = x.std(ddof=1)             # the usual slip: sample standard deviation
print("with ddof=1: sd %.4f -> standard scores %s" % (s1, ((new - mean) / s1).round(4)))

# ---- 49A.2
def psi(v, lam):                       # Eq. 49.3-6
    v = np.asarray(v, float); out = np.empty_like(v)
    p = v >= 0
    out[p] = np.log1p(v[p]) if lam == 0 else ((v[p] + 1) ** lam - 1) / lam
    q = ~p
    out[q] = -np.log1p(-v[q]) if lam == 2 else -((1 - v[q]) ** (2 - lam) - 1) / (2 - lam)
    return out
xs = np.array([-3.0, -1, 0, 1, 3])
for lam in (0, 2):
    print("49A.2  lambda", lam, " psi", psi(xs, lam).round(4), " scipy", stats.yeojohnson(xs, lmbda=lam).round(4))
    assert np.allclose(psi(xs, lam), stats.yeojohnson(xs, lmbda=lam))
print("terms: (1-x)^(2-lam) at x=-3, lam=0:", (1 + 3) ** 2, "; at x=-1:", (1 + 1) ** 2)
print("mirror: psi(x;2) = -psi(-x;0):", np.allclose(psi(xs, 2), -psi(-xs, 0)))

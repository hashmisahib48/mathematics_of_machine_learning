# Section 53.6: the two worked examples in NumPy and scikit-learn.
import numpy as np
from sklearn.feature_selection import SelectFdr, f_classif

# Example 53.6-1: ANOVA F (Eq. 53.3-1)
x = np.array([2., 3, 4, 3, 5, 7, 6, 8])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
grp = [x[y == k] for k in (0, 1)]
ssb = sum(len(v) * (v.mean() - x.mean()) ** 2 for v in grp)
ssw = sum(((v - v.mean()) ** 2).sum() for v in grp)
F = ssb / (ssw / (len(x) - 2))
Fs, Ps = f_classif(x[:, None], y)
assert np.isclose(F, Fs[0])
print("SSB %.2f SSW %.2f F %.2f p %.4f" % (ssb, ssw, F, Ps[0]))

# Example 53.6-2: Benjamini-Hochberg (Eq. 53.3-4), alpha = 0.05
p = np.array([0.044, 0.001, 0.032, 0.60, 0.008, 0.030])
m, alpha = len(p), 0.05
ps = np.sort(p)
line = alpha * np.arange(1, m + 1) / m
kstar = np.flatnonzero(ps <= line).max() + 1
sel = SelectFdr(lambda X, y: (np.zeros(m), p), alpha=alpha)
keep = sel.fit(np.zeros((9, m)), np.zeros(9)).get_support()
assert (keep == (p <= ps[kstar - 1])).all()
print("line", line.round(4), "k* =", kstar)
print("kept", keep.sum(), "| Bonferroni", (p < alpha / m).sum(),
      "| p < 0.05:", (p < alpha).sum())

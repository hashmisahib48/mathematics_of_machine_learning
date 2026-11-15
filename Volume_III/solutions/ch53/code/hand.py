# Hand checks for Problems 53A.1 - 53A.3 (Chapter 53 solutions)
import numpy as np
from scipy import stats
from sklearn.feature_selection import chi2, f_classif, SelectFdr, SelectFpr, SelectFwe

# 53A.1  three classes, three values each
x = np.arange(1., 10); y = np.repeat([0, 1, 2], 3)
means = [x[y == k].mean() for k in range(3)]; g = x.mean()
SSB = sum(3 * (m - g) ** 2 for m in means)
SSW = sum(((x[y == k] - means[k]) ** 2).sum() for k in range(3))
F = (SSB / 2) / (SSW / 6)
print("53A.1 means", means, "grand", g, "SSB", SSB, "SSW", SSW, "F", F)
print("      p (F dist)", stats.f.sf(F, 2, 6), "= (1+2F/6)^-3 =", (1 + 2 * F / 6) ** -3, " f_classif:", f_classif(x[:, None], y))
print("      F crit 5%% (2,6): %.3f" % stats.f.ppf(0.95, 2, 6))

# 53A.2  chi2, three classes of four documents
c = np.array([2, 2, 2, 2, 1, 1, 0, 0, 0, 1, 0, 1.]); z = np.repeat([0, 1, 2], 4)
O = np.array([c[z == k].sum() for k in range(3)]); E = c.sum() * 4 / 12
chi = ((O - E) ** 2 / E).sum()
print("53A.2 O", O, "E", E, "terms", (O - E) ** 2 / E, "chi2", chi, "p", round(stats.chi2.sf(chi, 2), 5))
print("      chi2():", chi2(c[:, None], z))

# 53A.3  eight p-values, alpha = 0.10
p = np.array([0.51, 0.002, 0.043, 0.20, 0.021, 0.08, 0.009, 0.034]); m = 8; al = 0.10
ps = np.sort(p); line = al * np.arange(1, m + 1) / m
print("53A.3 sorted", ps.tolist()); print("      line  ", line.round(4).tolist())
print("      under line:", (ps <= line).tolist(), " k* =", int(np.flatnonzero(ps <= line).max() + 1))
sel = lambda S: int(S(lambda X, yy: (np.zeros(m), p), alpha=al).fit(np.zeros((9, m)), np.zeros(9)).get_support().sum())
print("      Fdr", sel(SelectFdr), " Fpr", sel(SelectFpr), " Fwe", sel(SelectFwe), " Bonferroni limit", al / m)
print("      p <= 0.043 (BH cut):", sorted(p[p <= 0.043].tolist()))

# variation used in the instructor note of 53A.3: alpha = 0.05
al2 = 0.05; line2 = al2 * np.arange(1, m + 1) / m
print("53A.3 alpha=0.05: line", line2.round(5).tolist()); print("      under line:", (ps <= line2).tolist(), " k* =", int(np.flatnonzero(ps <= line2).max() + 1))
print("      SelectFdr(alpha=0.05) keeps", int(SelectFdr(lambda X, yy: (np.zeros(m), p), alpha=al2).fit(np.zeros((9, m)), np.zeros(9)).get_support().sum()))

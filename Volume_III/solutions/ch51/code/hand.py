# Hand calculations of the Chapter 51 solutions: Problems 51A.1 and 51A.2 (and instructor remarks)
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances

print("=== 51A.1  nan-Euclidean distances from row a")
nan = np.nan
M = np.array([[2, nan, 6], [4, 5, nan], [1, 3, 9], [nan, 6, 5]]); names = "abcd"; p = 3
for k in (1, 2, 3):
    both = ~np.isnan(M[0]) & ~np.isnan(M[k]); S = np.flatnonzero(both)
    ss = float(((M[0, S] - M[k, S]) ** 2).sum())
    print("a-%s: shared columns %s  sum of squares %g  x p/|S| = %g/%d  -> %.4f  -> distance %.3f"
          % (names[k], (S + 1).tolist(), ss, p, len(S), ss * p / len(S), np.sqrt(ss * p / len(S))))
print("sklearn:", nan_euclidean_distances(M[:1], M)[0].round(3).tolist())
donors = [1, 2, 3]; d = nan_euclidean_distances(M[:1], M)[0]
order = sorted(donors, key=lambda k: d[k]); print("donors for column 2 in order:", [names[k] for k in order], [M[k, 1] for k in order])
for k in (1, 2):
    print("k=%d fill = %.3f   (KNNImputer: %.3f)" % (k, np.mean([M[o, 1] for o in order[:k]]), KNNImputer(n_neighbors=k).fit_transform(M)[0, 1]))
print("distance-weighted k=2: %.3f" % ((M[order[0], 1] / d[order[0]] + M[order[1], 1] / d[order[1]]) / (1 / d[order[0]] + 1 / d[order[1]])))
print("without the p/|S| factor the distances would be:", [round(float(np.sqrt(((M[0] - M[k]) ** 2)[~np.isnan(M[0] - M[k])].sum())), 3) for k in (1, 2, 3)])

print("\n=== 51A.2  variance after a mean fill")
v = np.array([2, 4, 6, 8.]); no, n = 4, 6; m = v.mean(); so = ((v - m) ** 2).sum()
print("mean %g  sum of squares %g  s_o^2 = %g/%d = %g" % (m, so, so, no, so / no))
f = np.r_[v, m, m]; print("filled column", f.tolist(), " mean %g  sum of squares %g  variance %g/%d = %.4f" % (f.mean(), ((f - f.mean()) ** 2).sum(), ((f - f.mean()) ** 2).sum(), n, f.var()))
print("ratio %.4f   n_o/n = %.4f   with divisor 5 for the seen values and 5 for the filled: %.4f / %.4f" % (f.var() / (so / no), no / n, so / (no - 1), ((f - f.mean()) ** 2).sum() / (n - 1)))
q = 2 / 6; print("q = %.4f; correlation factor sqrt(1-q) = %.4f" % (q, np.sqrt(1 - q)))

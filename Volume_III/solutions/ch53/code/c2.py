# Problem 53C.2: how stable is the L1 selection of Section 53.9 under resampling?
import sys
sys.path.insert(0, '/home/claude/mlbook3/ch53/code')
import numpy as np
from bcdata import load
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

Xa, Xb, ya, yb, names = load(); Z = StandardScaler().fit_transform(Xa); n = len(ya); B = 40
l1 = lambda: SelectFromModel(LogisticRegression(l1_ratio=1, C=0.05, solver="liblinear", random_state=0))
full = np.flatnonzero(l1().fit(Z, ya).get_support()); print("selection on the full training rows:", full.tolist())
S = np.array([l1().fit(Z[i], ya[i]).get_support() for i in (np.random.RandomState(b).randint(0, n, n) for b in range(B))])
freq = S.mean(axis=0)
print("columns kept in all %d resamples:" % B, np.flatnonzero(freq == 1).tolist(), [str(names[j]) for j in np.flatnonzero(freq == 1)])
print("columns kept at least once:", int((freq > 0).sum()), " kept in at least 80 per cent:", int((freq >= 0.8).sum()))
print("size of the selected set: mean %.2f, min %d, max %d" % (S.sum(1).mean(), S.sum(1).min(), S.sum(1).max()))
print("the five columns of Section 53.9:")
for j in full: print("  column %2d %-22s kept in %2d of %d  (%.3f)" % (j, names[j], S[:, j].sum(), B, freq[j]))
print("exact copy of the full-sample set in %d of %d resamples" % (sum((np.flatnonzero(s) == full).all() if s.sum() == len(full) else False for s in S), B))
J = [(S[a] & S[b]).sum() / (S[a] | S[b]).sum() for a in range(B) for b in range(a + 1, B)]
print("mean Jaccard overlap of two resampled sets: %.3f (min %.3f)" % (np.mean(J), np.min(J)))
R = np.corrcoef(Z.T)
mid = [j for j in np.argsort(-freq) if 0 < freq[j] < 0.8][:4]
print("columns kept sometimes (0 < share < 0.8), strongest four:", [(int(j), round(float(freq[j]), 3)) for j in mid])
for j in mid:
    k = max(np.flatnonzero(freq == 1), key=lambda t: abs(R[j, t])); print("  column %2d: largest |r| with an always-kept column is %.3f (column %d)" % (j, abs(R[j, k]), k))
print("worst radius (20) and worst perimeter (22): r = %.3f; resamples with 20 dropped: %d, of which 22 kept: %d; with 20 kept, 22 kept in %d of %d" %
      (R[20, 22], (~S[:, 20]).sum(), (S[~S[:, 20], 22]).sum(), S[S[:, 20], 22].sum(), S[:, 20].sum()))
print("mean concavity (6) and mean concave points (7): r = %.3f" % R[6, 7])

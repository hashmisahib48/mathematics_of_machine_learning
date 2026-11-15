# Hand checks for the four class A problems of Chapter 42
import numpy as np
from sklearn.decomposition import PCA

print("== 42A.1  PCA of four points")
P = np.array([[6, 2], [3, 2], [5, 3], [6, 5]], float); n = len(P)
mu = P.mean(0); C = P - mu
print("mean", mu.tolist(), " centred rows", C.tolist())
print("sums of squares and products: Sxx = %.0f, Sxy = %.0f, Syy = %.0f" % ((C[:, 0] ** 2).sum(), (C[:, 0] * C[:, 1]).sum(), (C[:, 1] ** 2).sum()))
S = C.T @ C / (n - 1); print("S =", S.tolist())
lam, V = np.linalg.eigh(S); lam, V = lam[::-1], V[:, ::-1]
print("trace", np.trace(S), " det", round(np.linalg.det(S), 6), " eigenvalues", lam.round(6).tolist(), " (trace +- sqrt(trace^2/4 - det)):", 2 + 1, 2 - 1)
print("(S - 3I) w = 0 -> w = (1, 1)/sqrt2 = ", (np.array([1, 1]) / np.sqrt(2)).round(4).tolist(), " eigh:", np.abs(V[:, 0]).round(4).tolist())
w1 = np.array([1, 1]) / np.sqrt(2); w2 = np.array([-1, 1]) / np.sqrt(2)
z1, z2 = C @ w1, C @ w2
print("scores on PC1:", z1.round(4).tolist(), " sum of squares %.4f = (n-1) lambda1 = %.4f" % ((z1 ** 2).sum(), 3 * lam[0]))
print("scores on PC2:", z2.round(4).tolist(), " sum of squares %.4f = (n-1) lambda2 = %.4f" % ((z2 ** 2).sum(), 3 * lam[1]))
print("explained ratio %.4f ; sklearn:" % (lam[0] / lam.sum()), PCA(2).fit(P).explained_variance_ratio_.round(4).tolist(), " scores", PCA(1).fit_transform(P).ravel().round(4).tolist())
print("\n== 42A.2  how many components")
lam = np.array([4.2, 2.4, 1.6, 1.1, 0.7]); tot = lam.sum(); r = lam / tot; cs = np.cumsum(r)
print("total", round(tot, 4), " ratios", r.round(4).tolist(), " cumulative", cs.round(4).tolist())
for f in (0.8, 0.9, 0.95):
    k = int(np.argmax(cs > f)) + 1; print("f = %.2f: first k with cumulative share > f is k = %d (%.4f)" % (f, k, cs[k - 1]))
X = np.random.default_rng(0).normal(size=(2000, 5)) @ np.diag(np.sqrt(lam))     # data with these variances
pr = PCA().fit(X); print("sklearn on random data with these variances: ratios", pr.explained_variance_ratio_.round(2).tolist(),
      " components for 0.8, 0.9, 0.95:", [int(PCA(f).fit(X).n_components_) for f in (0.8, 0.9, 0.95)])
print("0.9: k = 3 gives 0.82 < 0.9, k = 4 gives 0.93 > 0.9 ; 0.95: k = 4 gives 0.93 < 0.95, so k = 5")

print("\n== 42A.3  error from the singular values")
sg = np.array([10, 6, 3, 1.]); n = 101
lam = sg ** 2 / (n - 1); print("sigma^2", (sg ** 2).tolist(), " eigenvalues", lam.tolist())
err = (sg[2:] ** 2).sum(); print("error of two components: sigma3^2 + sigma4^2 = %.0f ; (n-1)(l3 + l4) = %.0f" % (err, (n - 1) * lam[2:].sum()))
print("error per sample: %.4f ; total variance %.2f ; kept by two components %.4f ; discarded %.4f" % (err / n, lam.sum(), lam[:2].sum() / lam.sum(), lam[2:].sum() / lam.sum()))
print("per sample with divisor n-1 = 100: %.2f" % (err / 100))
print("slip: using sigma_j^2 / n instead of n-1 changes the eigenvalues to", (sg ** 2 / n).round(4).tolist(), "but the ratio is the same: %.4f" % ((sg[:2] ** 2).sum() / (sg ** 2).sum()))
# a real matrix with these singular values: n = 101 rows
rng = np.random.default_rng(1); Q, _ = np.linalg.qr(rng.normal(size=(101, 4))); V, _ = np.linalg.qr(rng.normal(size=(4, 4)))
Xc = Q @ np.diag(sg) @ V.T
print("check on a matrix with these singular values: Frobenius error of rank 2 =", round(float(np.linalg.norm(Xc - (Q[:, :2] * sg[:2]) @ V[:, :2].T) ** 2), 6))

print("\n== 42A.4  another fish")
F = np.array([[27, 4], [29, 6], [30, 6], [31, 8], [33, 6]], float); mu = F.mean(0); G = np.array([28., 6.])
w1 = np.array([2, 1]) / np.sqrt(5); w2 = np.array([-1, 2]) / np.sqrt(5)
c = G - mu; z1, z2 = c @ w1, c @ w2
back = mu + z1 * w1
print("centred G", c.tolist(), " w1 =", w1.round(4).tolist(), " w2 =", w2.round(4).tolist())
print("scores %.4f %.4f ; reconstruction" % (z1, z2), back.round(4).tolist(), "; squared error %.4f ; z2^2 = %.4f ; norm check %.4f + %.4f = %.4f" % (((G - back) ** 2).sum(), z2 ** 2, z1 ** 2, z2 ** 2, (c ** 2).sum()))
p = PCA(1).fit(F); print("sklearn:", p.transform(G[None]).round(4).tolist(), p.inverse_transform(p.transform(G[None])).round(4).tolist(), " components_", p.components_.round(4).tolist())
p2 = PCA(2).fit(F); print("sklearn 2 comps: scores", p2.transform(G[None]).round(4).tolist(), " components_[1]", p2.components_[1].round(4).tolist())
print("reconstruction error of the five fish for comparison: 0.2, 0.2, 0, 1.8, 1.8 ; mean", np.mean([0.2, 0.2, 0, 1.8, 1.8]))
print("slip: forgetting to centre: projection of (28, 6) on w1 gives %.4f instead of %.4f" % (G @ w1, z1))

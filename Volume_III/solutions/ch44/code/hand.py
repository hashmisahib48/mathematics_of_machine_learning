# Hand checks for Problems 44A.1 - 44A.4 (every number quoted in the solutions)
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import ClassicalMDS
from sklearn.manifold._locally_linear import barycenter_weights
np.set_printoptions(precision=4, suppress=True)

print("== 44A.1  villages P, Q, R with PQ = 6, PR = 8, QR = 10")
D = np.array([[0, 6, 8], [6, 0, 10], [8, 10, 0.]]); D2 = D ** 2; n = 3
print("squares:\n", D2)
r = D2.mean(1); g = D2.mean(); print("row means:", r, " = ", D2.sum(1), "/ 3;  overall mean:", g, " = ", D2.sum(), "/ 9")
B = -0.5 * (D2 - r[:, None] - r[None, :] + g); print("B =\n", B, "\n9 B =\n", 9 * B)
print("B_11 by hand: -1/2 (0 - 33.3333 - 33.3333 + 44.4444) =", -0.5 * (0 - r[0] - r[0] + g))
J = np.eye(n) - np.ones((n, n)) / n; print("same as -1/2 J D2 J:", np.allclose(B, -0.5 * J @ D2 @ J), "  row sums of B:", B.sum(1))
tr = np.trace(B); c2 = sum(B[i, i] * B[j, j] - B[i, j] ** 2 for i, j in ((0, 1), (0, 2), (1, 2)))
print("trace B =", tr, " sum of the 2x2 principal minors =", c2, " det B =", np.linalg.det(B))
disc = np.sqrt(tr ** 2 - 4 * c2); lam = np.array([(tr + disc) / 2, (tr - disc) / 2]); print("lambda^2 - trace*lambda + minors = 0: sqrt(disc) =", disc, " lambda =", lam)
w, V = np.linalg.eigh(B); w = w[::-1]; V = V[:, ::-1]; print("eigenvalues of B:", w)
Z = V[:, :2] * np.sqrt(w[:2])
for j in range(2): Z[:, j] *= np.sign(Z[np.abs(Z[:, j]).argmax(), j])          # largest entry positive, as ClassicalMDS
print("unit eigenvectors:", [(V[:, j] * np.sign(V[np.abs(V[:, j]).argmax(), j])).round(4).tolist() for j in range(2)])
print("coordinates (sqrt(lambda) v):\n", Z, "\nClassicalMDS:\n", ClassicalMDS(2, metric="precomputed").fit(D).embedding_)
print("distances of the map: PQ, PR, QR =", pdist(Z).round(4))
# the same eigenvalues from the centred true triangle P=(0,0), Q=(6,0), R=(0,8)
T = np.array([[0, 0], [6, 0], [0, 8.]]); Tc = T - T.mean(0); print("centred triangle:\n", Tc, "\n Tc^T Tc =\n", Tc.T @ Tc, " eigenvalues:", np.linalg.eigvalsh(Tc.T @ Tc)[::-1])

print("\n== 44A.2  weights of x = (2,1) from (0,0) and (5,0)")
x = np.array([2., 1.]); nb = np.array([[0, 0], [5, 0.]]); C = nb - x; G = C @ C.T
print("vectors to the neighbours:", C.tolist(), " G =\n", G, " det =", np.linalg.det(G), " G^-1 =\n", np.linalg.inv(G), "\n 25 G^-1 =\n", 25 * np.linalg.inv(G))
y = np.linalg.solve(G, np.ones(2)); print("G^-1 1 =", y, " sum", y.sum()); w = y / y.sum(); print("w =", w, " reconstruction", w @ nb, " error^2", ((x - w @ nb) ** 2).sum(), " w^T G w =", w @ G @ w)
print("terms of w^T G w: ", w[0]**2 * G[0, 0], 2 * w[0] * w[1] * G[0, 1], w[1]**2 * G[1, 1])
for reg in (0.01, 1e-3):
    R = reg * np.trace(G); Gr = G + R * np.eye(2); yr = np.linalg.solve(Gr, np.ones(2)); wr = yr / yr.sum()
    print(f"reg = {reg}: R = {R:.4f}  G+R I =\n", Gr, " det =", np.linalg.det(Gr), " G^-1 1 =", yr, " (x det:", yr * np.linalg.det(Gr), ")  w =", wr, " reconstruction", wr @ nb, " error^2", ((x - wr @ nb) ** 2).sum())
Gc = G + 0.01 * np.eye(2); yc = np.linalg.solve(Gc, np.ones(2)); print("constant 0.01 added without the trace:", yc / yc.sum())
print("scikit-learn barycenter_weights(reg=0.01):", barycenter_weights(x[None], nb, np.array([[0, 1]]), reg=0.01))

print("\n== 44A.3  perplexity")
def perp(p): p = np.array(p); return float(np.exp(-(p * np.log(p)).sum()))
for nm, p in (("a", [.5, .25, .25]), ("b", [.25] * 4), ("c", [.7, .2, .1])):
    p = np.array(p); H = -(p * np.log(p)).sum(); Hb = -(p * np.log2(p)).sum()
    print(f"({nm}) entropy {H:.4f} nats = {Hb:.4f} bits;  perplexity e^H = {np.exp(H):.4f} = 2^H_bits = {2**Hb:.4f}")
print("natural logarithm used with base 2:", 2 ** (-(np.array([.5, .25, .25]) * np.log(np.array([.5, .25, .25]))).sum()))
print("terms for (c) in nats:", [round(float(-q * np.log(q)), 4) for q in (.7, .2, .1)])
d2 = np.array([1., 4., 9.])
print("squared distances 1, 4, 9: perplexity as a function of sigma")
for s in (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
    p = np.exp(-d2 / (2 * s ** 2)); p /= p.sum(); print(f"  sigma {s:5.1f}: p = {p.round(4)}  perplexity {perp(p):.4f}")

print("\n== 44A.4  stress of a line")
d = np.array([3, 4, 5.]); z = np.array([0, 3, 7.]); e = np.array([abs(z[0] - z[1]), abs(z[0] - z[2]), abs(z[1] - z[2])])
print("map distances", e, " differences", d - e, " squares", (d - e) ** 2, " stress", ((d - e) ** 2).sum())
print("sum of squared map distances", (e ** 2).sum(), " Stress-1 =", np.sqrt(((d - e) ** 2).sum() / (e ** 2).sum()), " with the original distances in the denominator:", np.sqrt(((d - e) ** 2).sum() / (d ** 2).sum()))

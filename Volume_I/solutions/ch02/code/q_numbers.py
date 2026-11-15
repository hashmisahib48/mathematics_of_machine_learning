# Numbers quoted in the model answers to the questions of Chapter 2
import numpy as np
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("== Q1  word counts (3, 1), (6, 2), (1, 3)")
d1, d2, d3 = np.array([3., 1]), np.array([6., 2]), np.array([1., 3])
cs = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for n_, (a, b) in {"d1,d2": (d1, d2), "d1,d3": (d1, d3), "d2,d3": (d2, d3)}.items():
    print(f"{n_}: distance {np.linalg.norm(a - b):.3f}  cosine {cs(a, b):.4f}")
print("lengths:", np.linalg.norm(d1).round(3), np.linalg.norm(d2).round(3), np.linalg.norm(d3).round(3), " d2 = 2 d1:", np.allclose(d2, 2 * d1))

print("\n== Q2  first contact of a line a.w = t, moving outwards, with the unit balls")
th = rng.uniform(0, 2 * np.pi, 100000); a = np.column_stack([np.cos(th), np.sin(th)])
# the point of the ball that maximises a.w: L1 ball -> corner (sign(a_j) e_j with j = argmax |a_j|),
# L2 ball -> a itself, Linf ball -> (sign a1, sign a2)
j = np.abs(a).argmax(1); w1 = np.zeros_like(a); w1[np.arange(len(a)), j] = np.sign(a[np.arange(len(a)), j])
w2 = a; wi = np.sign(a)
tol = 1e-12
for name, w in (("L1 diamond", w1), ("L2 disc", w2), ("Linf square", wi)):
    print(f"{name:12s} share of contact points with one coordinate exactly 0: {np.mean(np.isclose(w, 0, atol=tol).any(1)):.4f}")
print("L1: the line touches an edge, not a corner, only when |a1| = |a2|: share with | |a1|-|a2| | < 1e-3:", np.mean(np.abs(np.abs(a[:, 0]) - np.abs(a[:, 1])) < 1e-3).round(4))

print("\n== Q3  a3 = a1 + a2")
A = rng.normal(size=(6, 2)); X = np.column_stack([A, A.sum(1)])
print("rank of X:", np.linalg.matrix_rank(X), " rank of X^T X:", np.linalg.matrix_rank(X.T @ X), " smallest singular value of X: %.1e" % np.linalg.svd(X, compute_uv=False)[-1])
y = rng.normal(size=6); w_ls = np.linalg.pinv(X) @ y; n0 = np.array([1., 1, -1])
print("X n0 =", np.round(X @ n0, 12), " (n0 = (1, 1, -1))")
for t in (0, 1, 10, -3.5): print(f"  t = {t:5}: |X(w + t n0) - X w| = {np.linalg.norm(X @ (w_ls + t * n0) - X @ w_ls):.1e}")
print("minimum-norm weights (pinv):", w_ls, " norm", np.linalg.norm(w_ls).round(4), "; with t = 1:", np.linalg.norm(w_ls + n0).round(4))

print("\n== Q4  the four matrices of Figure 2.5-1")
mats = {"scaling diag(2, 0.75)": np.diag([2, 0.75]), "rotation 30 deg": np.array([[np.cos(np.pi / 6), -np.sin(np.pi / 6)], [np.sin(np.pi / 6), np.cos(np.pi / 6)]]),
        "shear [[1,1],[0,1]]": np.array([[1., 1], [0, 1]]), "projection 0.5 everywhere": np.full((2, 2), 0.5)}
for k, M in mats.items():
    print(f"{k:28s} det = {np.linalg.det(M):7.4f}  rank = {np.linalg.matrix_rank(M)}")
    if abs(np.linalg.det(M)) > 1e-9: print("    inverse =", np.linalg.inv(M).round(4).tolist())
print("1/0.75 =", round(1 / 0.75, 4), " rotation angle of the inverse: -30 deg")

print("\n== Q5  normal equations against QR, correlation 0.999 (noise-free target, exact weights (1, 1))")
for rho in (0.9, 0.999):
    errs = []
    for rep in range(200):
        r = np.random.default_rng(rep); x1 = r.normal(size=50); x2 = rho * x1 + np.sqrt(1 - rho ** 2) * r.normal(size=50)
        X = np.column_stack([x1, x2]); yv = X @ np.ones(2)
        wn = np.linalg.solve(X.T @ X, X.T @ yv); Q, R = np.linalg.qr(X); wq = np.linalg.solve(R, Q.T @ yv)
        errs.append((np.linalg.norm(wn - 1), np.linalg.norm(wq - 1), np.linalg.cond(X), np.linalg.cond(X.T @ X)))
    e = np.median(np.array(errs), axis=0)
    print(f"rho = {rho}: median cond(X) {e[2]:.1f}, cond(X^T X) {e[3]:.0f};  error normal eq. {e[0]:.1e},  QR {e[1]:.1e}")
print("digits lost, log10(cond): X^T X at 1844.8 ->", np.log10(1844.8).round(2), "  X at 43.0 ->", np.log10(43.0).round(2))

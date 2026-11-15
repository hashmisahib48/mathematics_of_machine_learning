# Hand checks for the solutions of Chapter 46 (Problems 46A.1 - 46A.4 and numbers quoted for 46B.*)
import numpy as np
from scipy.stats import chi2
from sklearn.covariance import graphical_lasso
np.set_printoptions(precision=4, suppress=True)

print("== 46A.1  two points, one distance")
S = np.array([[4, 2], [2, 3.]]); Th = np.linalg.inv(S); print("det S", np.linalg.det(S), " 8 * Theta =", (8 * Th).ravel())
lam, V = np.linalg.eigh(S); print("eigenvalues of S", lam, " eigenvectors (columns)", V.ravel(), " correlation %.4f" % (2 / np.sqrt(12)))
for nm, q in (("A", [2, -1.]), ("B", [2, 1.])):
    q = np.array(q); z = V.T @ q
    print(nm, "Euclid^2", q @ q, " Theta q =", 8 * (Th @ q), " d^2 =", q @ Th @ q, " coordinates along the axes", z, " z^2/lambda", z**2 / lam, "sum", (z**2 / lam).sum(),
          " P(chi2_2 >= d^2) = %.4f" % chi2.sf(q @ Th @ q, 2))
for nm, q in (("A", [2, -1.]), ("B", [2, 1.])): print("slip: a^T S a for", nm, "=", np.array(q) @ S @ np.array(q))
print("97.5%% point of chi2_2: %.3f;  d^2 = 4 ellipse; half-axes sqrt(4 lambda): %s" % (chi2.isf(0.025, 2), np.sqrt(4 * lam)))

print("\n== 46A.2  shrinking two eigenvalues")
S = np.array([[5, 4], [4, 5.]]); lam = np.linalg.eigvalsh(S); mu = np.trace(S) / 2; a = 0.25
Sa = (1 - a) * S + a * mu * np.eye(2); la = np.linalg.eigvalsh(Sa)
print("eig S", lam, " cond", lam[1] / lam[0], " mu", mu, " shrunk\n", Sa, "\neig", la, " cond", la[1] / la[0], " alpha*mu =", a * mu)
print("each eigenvalue moves alpha of the way to mu:", lam + a * (mu - lam), "")
print("precision of S:", np.linalg.inv(S).ravel(), " eig", np.linalg.eigvalsh(np.linalg.inv(S)), "; of the shrunk:", np.linalg.inv(Sa).ravel(), " eig", np.linalg.eigvalsh(np.linalg.inv(Sa)))
print("trace S", np.trace(S), " trace shrunk", np.trace(Sa), "  alpha for condition number 1: 1")

print("\n== 46A.3  a chain")
T3 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2.]]); S3 = np.linalg.inv(T3)
print("det Theta", np.linalg.det(T3), " Sigma\n", S3, "\n4 Sigma\n", 4 * S3)
pc = lambda i, j: -T3[i, j] / np.sqrt(T3[i, i] * T3[j, j]); cr = lambda i, j: S3[i, j] / np.sqrt(S3[i, i] * S3[j, j])
print("partial 12 %.4f, 13 %.4f, 23 %.4f;  correlation 12 %.4f, 13 %.4f, 23 %.4f; product corr12 * corr23 = %.4f" % (pc(0, 1), pc(0, 2), pc(1, 2), cr(0, 1), cr(0, 2), cr(1, 2), cr(0, 1) * cr(1, 2)))
print("Sigma Theta = I:", np.allclose(S3 @ T3, np.eye(3)), "  regression of x1 on x2, x3: coefficients", -T3[0, 1:] / T3[0, 0], " residual variance", 1 / T3[0, 0])

print("\n== 46A.4  the largest useful penalty")
S4 = np.array([[1, 0.6, 0.2], [0.6, 1, 0.5], [0.2, 0.5, 1.]]); amax = np.abs(S4 - np.diag(np.diag(S4))).max(); print("alpha_max", amax)
print("alpha   Theta (upper triangle: 12, 13, 23)           edges  KKT: max |W_ij - S_ij| off the edges / on the edges")
for al in (0.7, 0.6, 0.55, 0.45, 0.3, 0.1):
    W, Th4 = graphical_lasso(S4, alpha=al, tol=1e-10, enet_tol=1e-10, max_iter=500)
    ed = np.abs(Th4[np.triu_indices(3, 1)]) > 1e-8; dev = np.abs(W - S4)[np.triu_indices(3, 1)]
    print("%5.2f   %s   %d    %s / %s" % (al, Th4[np.triu_indices(3, 1)].round(4), ed.sum(), dev[~ed].round(3), dev[ed].round(3)))
print("diag(1/S_ii) =", 1 / np.diag(S4))

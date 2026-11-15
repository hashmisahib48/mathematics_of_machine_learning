# Hand checks for Problems 21A.1 - 21A.3 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 21A.1  x = (2, 1), z = (1, 3), the map of Eq. 21.3-2")
def phi(v):
    a, b, s = v[0], v[1], np.sqrt(2)
    return np.array([1, s * a, s * b, a * a, s * a * b, b * b])
x = np.array([2., 1]); z = np.array([1., 3.])
print("phi(x) =", phi(x)); print("phi(z) =", phi(z)); pr = phi(x) * phi(z)
print("products:", pr, " sum:", pr.sum()); plain = lambda v: np.array([1, v[0], v[1], v[0] ** 2, v[0] * v[1], v[1] ** 2])
print("without the sqrt(2) factors: columns", plain(x), plain(z), " dot product =", plain(x) @ plain(z))
print("x.z =", x @ z, " (x.z + 1)^2 =", (x @ z + 1) ** 2)
print("sqrt(2) * 2 =", np.sqrt(2) * 2, " sqrt(2) * 3 =", np.sqrt(2) * 3)

print("\n== 21A.2  samples 1 and 2, targets 2 and 3, linear kernel, alpha = 1")
xs = np.array([1., 2.]); ys = np.array([2., 3.]); K = np.outer(xs, xs); A = K + np.eye(2)
print("K =", K.tolist(), " K + I =", A.tolist(), " det =", np.linalg.det(A))
a = np.linalg.solve(A, ys); print("a =", a, " (2/3, 1/3)")
# elimination as in the text: 2 a1 + 2 a2 = 2 ; 2 a1 + 5 a2 = 3
print("subtract the first equation from the second: 3 a2 =", 3 - 2, " so a2 =", 1 / 3, " a1 = (2 - 2 a2)/2 =", (2 - 2 / 3) / 2)
w = a @ xs; print("w = sum a_i x_i = %.4f   (2/3*1 + 1/3*2 = 4/3)" % w)
print("Eq. 10.3-6: w = x^T y / (x^T x + alpha) = %.4f / %.4f = %.4f" % (xs @ ys, xs @ xs + 1, (xs @ ys) / (xs @ xs + 1)))
print("f(3) = w * 3 =", 3 * w, "  by Eq. 21.4-4: sum a_i k(x_i, 3) =", sum(a * xs * 3))
print("fitted values K a =", K @ a, " y - alpha a =", ys - a)

print("\n== 21A.3  Laplacian kernel exp(-|x - z|) at x = 0, 1, 2")
X = np.array([0., 1, 2]); K = np.exp(-np.abs(X[:, None] - X[None])); print("K =\n", K)
c = np.array([1., -1, 1]); print("K c =", K @ c, "  c^T K c =", c @ K @ c, " = 3 - 2*2*e^-1 + 2*e^-2 =", 3 - 4 * np.exp(-1) + 2 * np.exp(-2))
e1, e2 = np.exp(-1), np.exp(-2)
print("eigenvector (1, 0, -1): K v =", K @ np.array([1., 0, -1]), " eigenvalue 1 - e^-2 =", 1 - e2)
# symmetric vectors (1, t, 1): matrix [[1+e2, e1], [2 e1, 1]]
M = np.array([[1 + e2, e1], [2 * e1, 1]]); tr, dt = np.trace(M), np.linalg.det(M)
print("symmetric block trace %.4f, det %.4f ; roots %.4f, %.4f" % (tr, dt, (tr + np.sqrt(tr ** 2 - 4 * dt)) / 2, (tr - np.sqrt(tr ** 2 - 4 * dt)) / 2))
print("eigvalsh:", np.linalg.eigvalsh(K), " product of eigenvalues =", np.prod(np.linalg.eigvalsh(K)), " det K =", np.linalg.det(K))
print("det K = (1 - e^-2)^2 = %.4f ; product of the two symmetric-block roots = %.4f = 1 - e^-2" % ((1 - e2) ** 2, 1.5923 * 0.5430))
print("leading minors of K: 1, 1 - e^-2 = %.4f, det K = %.4f" % (1 - e2, np.linalg.det(K)))

# FastICA by hand on the two-microphone toy (Eqs. 43.4-2 to 43.4-5)
import numpy as np
from sklearn.decomposition import FastICA

S = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], float)  # sources
A = np.array([[2., 1.], [1., 1.]])                         # mixing
X = S @ A.T                                    # microphones, 4 x 2
C = X.T @ X / 4                                # covariance, mean 0
lam, E = np.linalg.eigh(C)
lam, E = lam[::-1], E[:, ::-1]                  # descending
E = E * np.sign(E[1])                            # fix the signs
K = (E / np.sqrt(lam)).T                         # whitening matrix
Z = X @ K.T                                      # whitened data
print("eigenvalues", lam.round(4), " cov(Z) = I:",
      np.allclose(Z.T @ Z / 4, np.eye(2)))
for deg in (0, 30, 60, 90, 120, 150):            # kurtosis of w'z
    t = np.radians(deg); y = Z @ [np.cos(t), np.sin(t)]
    print(f"angle {deg:3d}  E[y^4] = {np.mean(y**4):.4f}")
w = np.array([1.0, 0.0])                         # start; g(u) = u^3
for it in range(1, 5):
    y = Z @ w
    w = (Z * (y**3)[:, None]).mean(axis=0) - 3 * w   # Eq. 43.4-5
    w /= np.linalg.norm(w)
    print(it, "w =", w.round(4), " y =", (Z @ w).round(4))
w2 = np.array([-w[1], w[0]])                     # orthogonal row
Wk = np.vstack([w, w2]) @ K                      # total unmixing
print("W K =\n", Wk.round(4), "\nA^-1 =\n", np.linalg.inv(A))
ica = FastICA(2, fun="cube", whiten="unit-variance", random_state=0)
Sh = ica.fit_transform(X)
print("sklearn sources (columns) =\n", Sh.round(4))

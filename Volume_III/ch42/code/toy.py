# Five fish: PCA by hand, by the covariance and by the SVD.
import numpy as np

# length (cm) and weight (hundreds of grams) of five fish
X = np.array([[27, 4], [29, 6], [30, 6], [31, 8], [33, 6]], float)
n = len(X)
mu = X.mean(axis=0)
Xc = X - mu                              # centred data
S = Xc.T @ Xc / (n - 1)                  # Eq. 42.3-1
lam, W = np.linalg.eigh(S)               # ascending order
lam, W = lam[::-1], W[:, ::-1]           # largest first
big = np.abs(W).argmax(axis=0)           # sign rule of svd_flip:
W = W * np.sign(W[big, [0, 1]])          # largest entry positive
print("mean", mu, " S =", S.tolist())
print("eigenvalues", lam.round(4), "ratio",
      (lam / lam.sum()).round(4))
print("components\n", W.T.round(4))
Z = Xc @ W                               # scores, Eq. 42.4-3
print("scores\n", Z.round(4))
R = Xc - np.outer(Z[:, 0], W[:, 0])      # residual of 1 component
print("residual SS", round(float((R**2).sum()), 4),
      " (n-1)*lambda_2 =", round((n - 1) * lam[1], 4))

U, s, Vt = np.linalg.svd(Xc, full_matrices=False)    # Eq. 42.4-1
print("singular values", s.round(4),
      " s^2/(n-1)", (s**2 / (n - 1)).round(4))
Zw = Z / np.sqrt(lam)                    # whitened scores
print("whitened variances", Zw.var(axis=0, ddof=1).round(4))
new = np.array([32.0, 5.0])              # a sixth fish
z = (new - mu) @ W
back = mu + z[0] * W[:, 0]
err = float(((new - back) ** 2).sum())
print("new fish scores", z.round(4), " back", back.round(4))
print("error^2", round(err, 4))

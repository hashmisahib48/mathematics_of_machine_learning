import numpy as np

colour = np.array(list("rrrbbwww"))      # red, blue, white
price = np.array([15, 12, 9, 10, 8, 9, 6, 3.])
cats = np.unique(colour)                  # sorted: b, r, w
D = (colour[:, None] == cats).astype(float)
A = np.hstack([np.ones((8, 1)), D])
print("rank [1|D] =", np.linalg.matrix_rank(A), "for", A.shape[1],
      "columns")
n_c, ybar = D.sum(0), price.mean()
mean_c = D.T @ price / n_c                # Eq. 50.3-1
print("n_c", n_c, "mean_c", mean_c, "ybar", ybar)
t = (n_c * mean_c + 2 * ybar) / (n_c + 2)  # Eq. 50.4-1, m = 2
print("m=2 encoding:", t.round(3))
var_c = np.array([price[colour == c].var() for c in cats])
lam = n_c / (n_c + var_c / price.var())   # Eq. 50.4-2
print("auto encoding:", (lam * mean_c + (1 - lam) * ybar).round(3))

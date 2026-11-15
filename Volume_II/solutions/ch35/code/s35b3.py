# Problem 35B.3: PLSRegression is least squares on its scores.
import sys
sys.path.append("../../../ch35/code")  # chapter data
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from spectra import make_spectra

X, y = make_spectra(60, 0)
pls = PLSRegression(3).fit(X, y)
T = pls.x_scores_                          # 60 x 3 scores
ols = LinearRegression().fit(T, y)         # y on the three scores

diff = np.abs(ols.predict(T) - pls.predict(X).ravel()).max()
print(f"largest difference of the predictions: {diff:.1e}")
print(f"intercepts: {ols.intercept_:.4f} and {pls.intercept_[0]:.4f}")
sy = y.std(ddof=1)
print("LinearRegression coef_ :", ols.coef_.round(5))
print("y_loadings_ x sd(y)    :",
      (pls.y_loadings_.ravel() * sy).round(5))
print("squared norms of scores:", (T ** 2).sum(0).round(2))
G = T.T @ T                                # Gram matrix of the scores
off = np.abs(G - np.diag(np.diag(G))).max()
print(f"largest t_i . t_j, i != j: {off:.1e}")
print("corr(t1, t2) = %.1e" % np.corrcoef(T[:, 0], T[:, 1])[0, 1])
# each coefficient alone, since the scores are orthogonal
yc = y - y.mean()
q = (T.T @ yc) / (T ** 2).sum(0)
print("t_k . (y - mean) / t_k . t_k:", q.round(5))

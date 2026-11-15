# The nu property and the dual (Eq. 48.4-7, 48.4-8).
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics.pairwise import rbf_kernel
from sensors import make_sensors

X, y = make_sensors()
Z = StandardScaler().fit_transform(X)
n = len(Z)
print(" nu  outliers  at bound   SVs  sum(alpha)  nu*n")
for nu in (0.01, 0.05, 0.1, 0.2, 0.5):
    m = OneClassSVM(nu=nu, tol=1e-6).fit(Z)
    a = m.dual_coef_[0]                      # alpha' = nu n alpha
    out = (m.decision_function(Z) < -1e-6).mean()
    print(f"{nu:4.2f} {out:8.3f} {(a > 1 - 1e-8).sum() / n:8.3f} "
          f"{len(a) / n:6.3f} {a.sum():11.3f} {nu * n:6.2f}")
m = OneClassSVM(nu=0.1, tol=1e-6).fit(Z)
g = 1 / (Z.shape[1] * Z.var())                   # gamma = scale
K = rbf_kernel(Z, m.support_vectors_, gamma=g)
f = K @ m.dual_coef_[0] - m.offset_[0]           # Eq. 48.4-8
print("decision_function = sum_i alpha_i k(x_i, x) - rho:",
      np.allclose(f, m.decision_function(Z)))
print("gamma_ = 1/(p Var X) =", round(g, 4),
      "; rho =", round(float(m.offset_[0]), 4))

# Problem 21B.2: leave-one-out errors without refitting, Eq. 21.4-7
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

rng = np.random.default_rng(4)
x = np.sort(rng.uniform(0, 6, 28))
y = np.sin(x) + 0.25 * rng.normal(size=28)
gamma, alpha = 0.1, 0.03
K = rbf_kernel(x[:, None], gamma=gamma)
H = K @ np.linalg.inv(K + alpha * np.eye(28))      # y_hat = H y
resid = y - H @ y                                  # residuals
loo_formula = resid / (1 - np.diag(H))             # Eq. 21.4-7

loo_refit = np.zeros(28)
for i in range(28):                               # 28 explicit fits
    keep = np.arange(28) != i
    m = KernelRidge(alpha=alpha, kernel="rbf", gamma=gamma)
    m.fit(x[keep, None], y[keep])
    loo_refit[i] = y[i] - m.predict(x[i:i + 1, None])[0]

rmse = lambda e: np.sqrt(np.mean(e ** 2))
d = abs(loo_formula - loo_refit).max()
print("largest difference: %.1e" % d)
print("df = trace(H) = %.3f   (average H_ii = %.4f)"
      % (np.trace(H), np.diag(H).mean()))
print("RMSE of ordinary residuals    %.4f" % rmse(resid))
print("RMSE of leave-one-out errors  %.4f" % rmse(loo_refit))
fac = 1 / (1 - np.diag(H))
j = fac.argmax()
print("factor 1/(1 - H_ii): smallest %.3f, largest %.3f"
      % (fac.min(), fac.max()))
print("   largest at x = %.2f, where H_ii = %.3f" % (x[j], H[j, j]))
big = np.argsort(-abs(loo_refit))[:3]
print("largest leave-one-out errors at x =", x[big].round(2))
print("   H_ii there:", np.diag(H)[big].round(3))

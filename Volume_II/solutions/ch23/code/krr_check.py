# Evidence for the note to 23B.2: KernelRidge(alpha=0.1, gamma=0.5) as written in the statement has the default linear kernel.
import numpy as np
from sklearn.kernel_ridge import KernelRidge
rng = np.random.default_rng(1); x = rng.uniform(0, 6, 300); y = np.sin(x) + 0.25 * rng.normal(size=300); X = x[:, None]
rm = lambda p, t: np.sqrt(np.mean((p - t)**2))
for nm, k, c in (("as written: KernelRidge(alpha=0.1, gamma=0.5)", KernelRidge(alpha=0.1, gamma=0.5), False),
                 ("kernel='rbf', y not centred", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5), False),
                 ("kernel='rbf', y centred", KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5), True)):
    m = y[:200].mean() if c else 0
    k.fit(X[:200], y[:200] - m); print("%-48s test RMSE %.4f" % (nm, rm(k.predict(X[200:]) + m, y[200:])))

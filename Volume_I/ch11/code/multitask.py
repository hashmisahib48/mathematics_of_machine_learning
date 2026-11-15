# MultiTaskLasso: one feature by hand, then the Linnerud data.
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, MultiTaskLasso
from sklearn.preprocessing import StandardScaler

x = np.array([-1.0, -1.0, 1.0, 1.0])           # v = 1, centred
Y = np.c_[3 * x, 4 * x]                        # z = (3, 4), |z| = 5
for a in (2.0, 3.5):
    mt = MultiTaskLasso(alpha=a, tol=1e-12).fit(x[:, None], Y)
    each = [Lasso(alpha=a, tol=1e-12).fit(x[:, None], Y[:, k])
            .coef_[0] for k in (0, 1)]
    shrink = max(1 - a / 5.0, 0.0)             # Eq. 11.12-3
    print(f"alpha={a}: multi-task", np.round(mt.coef_.ravel(), 4),
          "formula", shrink * np.array([3, 4]))
    print("   separate lassos", np.round(each, 4) + 0.0)

data = load_linnerud()                         # 20 people
X = StandardScaler().fit_transform(data.data)  # exercises
Y = StandardScaler().fit_transform(data.target)  # body data
alpha = 0.4
mt = MultiTaskLasso(alpha=alpha).fit(X, Y)
print("\ncoef_ of MultiTaskLasso, shape", mt.coef_.shape)
print("            Chins  Situps  Jumps")
for name, row in zip(data.target_names, mt.coef_):
    print(f"  {name:8s}", np.round(row, 3) + 0.0)
print("separate lassos, same alpha")
for k, name in enumerate(data.target_names):
    w = Lasso(alpha=alpha).fit(X, Y[:, k]).coef_
    print(f"  {name:8s}", np.round(w, 3) + 0.0)

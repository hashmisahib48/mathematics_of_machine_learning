# One-weight ridge: the penalty shrinks the slope (Eq. 6.5-3).
import numpy as np
from sklearn.linear_model import Ridge

x = np.array([-2.0, -1.0, 1.0, 2.0])         # centred feature
y = np.array([-4.0, 0.0, 1.0, 3.0])          # centred target
x_new = np.array([-3.0, -1.0, 2.0, 3.0])     # test samples
y_new = np.array([-2.5, -1.5, 1.0, 4.0])

S_xy, S_xx = np.sum(x * y), np.sum(x * x)
print(f"S_xy = {S_xy:.0f}   S_xx = {S_xx:.0f}")
print("alpha   w    train MSE  test MSE  scikit-learn w")
for alpha in (0, 5, 10, 20, 40):
    w = S_xy / (S_xx + alpha)                # Eq. 6.5-3
    tr = np.mean((y - w * x) ** 2)
    te = np.mean((y_new - w * x_new) ** 2)
    ridge = Ridge(alpha=alpha).fit(x[:, None], y)
    print(f"{alpha:5d} {w:6.3f} {tr:9.3f} {te:9.3f} "
          f"{ridge.coef_[0]:12.3f}")

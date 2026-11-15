# The best constant prediction for five waiting times (Eq. 1.4-4).
import numpy as np
y = np.array([2.0, 3.0, 4.0, 5.0, 16.0])        # minutes
def mse(c):                     # Eq. 1.4-4
    return np.mean((y - c) ** 2)
def mae(c):                     # mean absolute loss of c
    return np.mean(np.abs(y - c))
print("   c     MSE     MAE")
for c in (2.0, 3.0, 4.0, 5.0, 6.0):
    print(f"{c:5.1f} {mse(c):7.2f} {mae(c):7.2f}")
grid = np.round(np.arange(0.0, 20.0, 0.01), 2)   # candidates
J_sq = np.array([mse(c) for c in grid])
J_abs = np.array([mae(c) for c in grid])
print("grid minimum, squared loss  :", grid[J_sq.argmin()])
print("grid minimum, absolute loss :", grid[J_abs.argmin()])
print("mean and median of y        :", y.mean(), np.median(y))
print("MSE at the mean = variance  :", mse(y.mean()), y.var())

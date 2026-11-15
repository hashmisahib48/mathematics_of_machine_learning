# Problem 1B.1: an outlier in the list of waiting times
import numpy as np
y = np.array([2.0, 3.0, 4.0, 5.0, 16.0, 100.0])   # Listing 1.1 + 100

def mse(c):                     # Eq. 1.4-4
    return np.mean((y - c) ** 2)
def mae(c):                     # mean absolute loss
    return np.mean(np.abs(y - c))

grid = np.round(np.arange(0.0, 30.0, 0.01), 2)    # candidates 0..30
J_sq = np.array([mse(c) for c in grid])
J_abs = np.array([mae(c) for c in grid])
print("grid minimum, squared loss :", grid[J_sq.argmin()])
print("grid minimum, absolute loss:", grid[J_abs.argmin()])
print("mean, median of y :", round(y.mean(), 3), np.median(y))
print("MSE at the mean   :", round(mse(y.mean()), 2))
print("MAE at the median :", round(mae(np.median(y)), 3))

flat = grid[np.isclose(J_abs, J_abs.min(), atol=1e-9)]
print("flat bottom of MAE:", flat.min(), "to", flat.max())
vals = [round(float(mae(c)), 3) for c in (4, 4.5, 5)]
print("MAE at 4, 4.5, 5  :", vals)
slope = (mae(4.51) - mae(4.49)) / 0.02
print("slope at c = 4.5   :", round(float(slope), 6))
# Without the outlier (Listing 1.1): mean 6 and median 4
y5 = y[:5]
print("without 100       :", y5.mean(), np.median(y5))

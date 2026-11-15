# The delivery data set used throughout Chapter 9.
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])    # distance, km
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])   # delivery time, min

# Slope and intercept from the two formulas of Eq. 9.3-5.
x_bar, y_bar = x.mean(), y.mean()
S_xy = np.sum((x - x_bar) * (y - y_bar))
S_xx = np.sum((x - x_bar) ** 2)
w = S_xy / S_xx
b = y_bar - w * x_bar

y_hat = b + w * x             # predictions
r = y - y_hat                 # residuals
SSE = np.sum(r ** 2)          # what least squares minimises
SST = np.sum((y - y_bar) ** 2)

print(f"S_xy = {S_xy:.1f}   S_xx = {S_xx:.1f}")
print(f"w = {w:.2f}   b = {b:.2f}")
print("y_hat =", np.round(y_hat, 2))
print("r     =", np.round(r, 2))
print(f"SSE = {SSE:.2f}   SST = {SST:.1f}   R2 = {1 - SSE/SST:.4f}")

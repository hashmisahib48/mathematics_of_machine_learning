# Gradient descent for the best constant (Eq. 1.5-1 to 1.5-3).
import numpy as np
y = np.array([2.0, 3.0, 4.0, 5.0, 16.0])
y_bar = y.mean()
def grad(c):                    # Eq. 1.5-2, slope at c
    return -2.0 * np.mean(y - c)
def descend(eta, c=0.0, steps=5):
    path = [c]
    for _ in range(steps):
        c = c - eta * grad(c)   # Eq. 1.5-1: step against the slope
        path.append(c)
    return np.array(path)
print("mean of y =", y_bar)
for eta in (0.25, 0.9, 1.1):
    print(f"eta = {eta:4.2f}  factor 1-2*eta = {1 - 2 * eta:5.2f}")
    print("   c_t =", np.round(descend(eta), 4))
path = descend(0.25)
print("J(c_t) for eta = 0.25:", np.round(
    [np.mean((y - c) ** 2) for c in path], 4))

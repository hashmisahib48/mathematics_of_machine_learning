# Newton's method: a curved function in 1-D, a bowl in 2-D.
import numpy as np

sig = lambda w: 1 / (1 + np.exp(-w))
f1 = lambda w: sig(w) - 0.3                # f'(w) for
f2 = lambda w: sig(w) * (1 - sig(w))       # f = ln(1+e^w) - 0.3 w
w_star = np.log(0.3 / 0.7)
print(f"minimum at w* = {w_star:.6f}")
wn = wg = 0.0
print(" t    Newton w   error      gradient w   error")
for t in range(1, 6):
    wn = wn - f1(wn) / f2(wn)              # Eq. 5.5-2
    wg = wg - f1(wg)                       # eta = 1
    print(f"{t:2d} {wn:11.6f} {wn - w_star:+.1e} {wg:11.6f}"
          f" {wg - w_star:+.1e}")

x = np.arange(1.0, 6.0)
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]
w0 = np.zeros(2)
grad = -2 * X.T @ (y - X @ w0)             # Eq. 5.3-3
H = 2 * X.T @ X                            # Hessian, constant
print("gradient at (0, 0):", grad, " H =", H.tolist())
step = np.linalg.solve(H, grad)            # solve H d = g, not H^-1
print("step d:", step.round(4), "  new w:", (w0 - step).round(4))

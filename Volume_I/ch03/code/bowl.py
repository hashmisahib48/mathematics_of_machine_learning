# The loss of the delivery data as a function of (b, w).
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
J = lambda b, w: np.sum((y - b - w * x) ** 2)

# One variable (Example 3.3-1): fix b = 0.9, J is a parabola in w.
a2 = np.sum(x * x)                         # coefficient of w^2
a1 = -2 * np.sum(x * (y - 0.9))            # coefficient of w
a0 = np.sum((y - 0.9) ** 2)
print(f"J(w) = {a2:.0f} w^2 {a1:+.0f} w {a0:+.2f}")
w_star = -a1 / (2 * a2)                    # J'(w) = 0
print(f"J' = 0 at w = {w_star:.4f}")
print(f"J'' = {2*a2:.0f}   J at the minimum = {J(0.9, w_star):.2f}")

# Two variables (Example 3.5-1): the gradient at (b, w) = (0, 1).
b0, w0 = 0.0, 1.0
r = y - b0 - w0 * x
g = np.array([-2 * r.sum(), -2 * np.sum(x * r)])  # Eq. 3.5-3
h = 1e-6
g_num = np.array([(J(b0 + h, w0) - J(b0 - h, w0)) / (2 * h),
                  (J(b0, w0 + h) - J(b0, w0 - h)) / (2 * h)])
print("residuals:", r, "  J =", J(b0, w0))
print("gradient:", g, "  numeric:", np.round(g_num, 4))
print(f"length of gradient = {np.linalg.norm(g):.2f}")

# Take a tiny step uphill, downhill and along the contour.
t = 1e-4
u = g / np.linalg.norm(g)                  # unit uphill direction
c = np.array([-u[1], u[0]])                # turned by 90 deg
moves = [("uphill", u), ("downhill", -u), ("along contour", c)]
for name, d in moves:
    dJ = J(b0 + t * d[0], w0 + t * d[1]) - J(b0, w0)
    print(f"{name:<14} change of J per unit step {dJ / t:+9.3f}")

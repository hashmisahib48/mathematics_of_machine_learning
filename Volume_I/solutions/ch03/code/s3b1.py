# Problem 3B.1: check of the ridge gradient (delivery data)
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])
X = np.c_[np.ones(5), x]           # columns 1 and x; v = (b, w)
alpha = 2.0

def num_grad(f, v, h=1e-6):
    """Central differences, one coordinate at a time."""
    g = np.zeros_like(v)
    for j in range(v.size):
        e = np.zeros_like(v); e[j] = h
        g[j] = (f(v + e) - f(v - e)) / (2 * h)     # Eq. 3.10-1
    return g

def rel(a, b):                     # relative error, Eq. 3.10-2
    return np.linalg.norm(a - b) / (np.linalg.norm(a)
                                    + np.linalg.norm(b))

J = lambda v: np.sum((y - X @ v) ** 2) + alpha * v @ v
gJ = lambda v: -2 * X.T @ (y - X @ v)              # Eq. 3.7-6
grad = lambda v: gJ(v) + 2 * alpha * v             # right penalty
wrong = lambda v: gJ(v) + alpha * v                # forgot the 2

for v in (np.array([0.0, 1.0]), np.array([1.0, 2.0])):
    num = num_grad(J, v)
    print("v =", v)
    print("  gradient of J     :", gJ(v))
    print("  penalty 2*alpha*v :", 2 * alpha * v)
    print("  analytic total    :", grad(v))
    print("  numeric           :", np.round(num, 4))
    print("  wrong penalty     :", wrong(v))
    print(f"  rel. error right {rel(grad(v), num):.1e}"
          f"   wrong {rel(wrong(v), num):.1e}")

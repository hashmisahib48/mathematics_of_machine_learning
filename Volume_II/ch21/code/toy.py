# The three-point data set and the hand calculations (Section 21.5).
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel


def phi(v):                      # Eq. 21.3-2: degree-2 map in 2-D
    a, b, s = v[0], v[1], np.sqrt(2)
    return np.array([1, s * a, s * b, a * a, s * a * b, b * b])


x, z = np.array([1.0, 2.0]), np.array([3.0, 1.0])
print("phi(x).phi(z) =", round(float(phi(x) @ phi(z)), 4))
print("(x.z + 1)^2   =", float((x @ z + 1) ** 2))

X = np.array([[-1.0], [0.0], [1.0]])
y = np.array([1.0, 0.0, 1.0])
alpha, I = 0.1, np.eye(3)
kernels = {
    "linear": X @ X.T,
    "poly, degree 2": polynomial_kernel(X, degree=2, gamma=1),
    "rbf, gamma 0.5": rbf_kernel(X, gamma=0.5),
}
for name, K in kernels.items():
    a = np.linalg.solve(K + alpha * I, y)          # Eq. 21.4-3
    print(f"{name:15s} a = {np.round(a, 4)}")
    print(f"{'':15s} f(data) = {np.round(K @ a, 4)}")
    if name.startswith("rbf"):
        g, x_new = 0.5, np.array([[0.5], [4.0]])
        k_new = np.exp(-g * (x_new - X.T) ** 2)    # 2 x 3 matrix
        print(f"{'':15s} f(0.5), f(4) = {np.round(k_new @ a, 4)}")

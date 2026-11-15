# Determinant, inverse and solving a system (Section 2.7).
import numpy as np

K = np.array([[4.0, 2.0], [3.0, 2.0]])
b = np.array([10.0, 8.0])
print("det K =", round(float(np.linalg.det(K)), 6))
print("inv K =", np.linalg.inv(K).tolist())       # Eq. 2.7-3
print("solve =", np.linalg.solve(K, b))           # Eq. 2.7-1
print("K inv(K) = I:", np.allclose(K @ np.linalg.inv(K), np.eye(2)))

S = np.array([[2.0, 4.0], [1.0, 2.0]])          # row 1 = 2 x row 2
print("det S =", np.linalg.det(S),
      "  rank S =", np.linalg.matrix_rank(S))
try:
    np.linalg.solve(S, b)
except np.linalg.LinAlgError as e:
    print("LinAlgError:", e)

Q = np.array([[0.0, 1.0], [1.0, 0.0]])          # swaps the axes
print("det Q =", np.linalg.det(Q))               # orientation flips

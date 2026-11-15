# QR factorisation of A (Section 2.8).
import numpy as np

A = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
Q, R = np.linalg.qr(A)                          # Eq. 2.8-3
print("Q =\n", np.round(Q, 4) + 0.0)
print("R =\n", np.round(R, 4) + 0.0)
print("Q^T Q = I:", np.allclose(Q.T @ Q, np.eye(2)),
      "  QR = A:", np.allclose(Q @ R, A))

# Hand computation of Eq. 2.8-4, to compare with the library.
a1, a2 = A[:, 0], A[:, 1]
q1 = a1 / np.linalg.norm(a1)
v2 = a2 - (q1 @ a2) * q1
q2 = v2 / np.linalg.norm(v2)
print("q1 =", np.round(q1, 4), "  q2 =", np.round(q2, 4))
r = [np.linalg.norm(a1), q1 @ a2, np.linalg.norm(v2)]
print("r11, r12, r22 =", np.round(r, 4))

x = np.array([3.0, -1.0])
print("|x| =", round(float(np.linalg.norm(x)), 4),
      "  |Rx| =", round(float(np.linalg.norm(R @ x)), 4))

y = np.array([3.0, 3.0, 3.0])                  # least squares by QR
print("solve R c = Q^T y:", np.linalg.solve(R, Q.T @ y))

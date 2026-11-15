import numpy as np
from sklearn.metrics import mutual_info_score

A = np.array([[2., 1], [1, 0], [0, 1]])
U, s, Vt = np.linalg.svd(A, full_matrices=False)       # Eq. R.2-1
A1 = s[0] * np.outer(U[:, 0], Vt[0])                   # best rank 1
print("sigma:", s.round(4), " ||A-A1||_F:",
      round(np.linalg.norm(A - A1), 4))
P = np.array([[0.4, 0.1], [0.1, 0.4]])                 # joint table
Q = np.outer(P.sum(1), P.sum(0))                       # margins
print("MI = KL(P||Q):", round(np.sum(P * np.log(P / Q)), 4),
      " sklearn:", round(mutual_info_score(
          None, None, contingency=P * 100), 4))        # Eq. R.4-3
def kl(m1, v1, m2, v2):                                # Eq. R.4-4
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)
print("KL(N(0,1)||N(1,4)):", round(kl(0, 1, 1, 4), 4),
      " reverse:", round(kl(1, 4, 0, 1), 4))

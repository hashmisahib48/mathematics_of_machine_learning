# Example 44.8-2: reconstruction weights of one point.
import numpy as np
from sklearn.manifold._locally_linear import barycenter_weights

np.set_printoptions(precision=4, suppress=True)
x = np.array([[1.0, 1.0]])                      # the point
nb = np.array([[0.0, 0.0], [3.0, 0.0]])         # its two neighbours
C = nb - x                                      # rows x_j - x_i
G = C @ C.T                                     # local Gram matrix
print("G =", G.tolist(), " trace =", np.trace(G))
for reg in (0.0, 1e-3):
    R = reg * np.trace(G)                       # as in scikit-learn
    z = np.linalg.solve(G + R * np.eye(2), np.ones(2))
    w = z / z.sum()                             # Eq. 44.5-3
    rec = w @ nb
    err = np.sum((x[0] - rec) ** 2)
    print(f"reg={reg}: w =", w, "reconstruction", rec,
          "squared error", round(float(err), 6))
ref = barycenter_weights(x, nb, np.array([[0, 1]]), reg=1e-3)
print("scikit-learn (reg=1e-3):", ref[0])

# three neighbours in the plane: G is singular
nb3 = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
C3 = nb3 - x
G3 = C3 @ C3.T
print("rank of G with 3 neighbours:", np.linalg.matrix_rank(G3))
w3 = barycenter_weights(x, nb3, np.array([[0, 1, 2]]), reg=1e-3)[0]
print("regularised weights:", w3, "sum", w3.sum().round(6))
print("reconstruction:", (w3 @ nb3).round(4))

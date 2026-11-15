# The five-node chain of Section 34.1 (Examples 34.5-1 and 34.5-2).
import numpy as np
np.set_printoptions(precision=4, suppress=True)

W = np.zeros((5, 5))                    # similarity of neighbours
for i, w in enumerate([2, 4, 1, 1]):
    W[i, i + 1] = W[i + 1, i] = w
d = W.sum(axis=1)                       # degrees, Eq. 34.3-3
L, U = [0, 4], [1, 2, 3]                # labelled, unlabelled nodes
fL = np.array([0.0, 1.0])               # class 0 -> 0, class 1 -> 1

# Harmonic solution, Eq. 34.3-6
A = np.diag(d[U]) - W[np.ix_(U, U)]
fU = np.linalg.solve(A, W[np.ix_(U, L)] @ fL)
f = np.zeros(5)
f[L], f[U] = fL, fU
print("degrees          :", d)
print("f_U              :", fU, " 11 f_U =", np.round(11 * fU, 6))
print("weighted mean    :", (W @ f / d)[U])

# Iteration, Eq. 34.3-7: multiply by T = D^-1 W, clamp the labels
T = W / d[:, None]
g = np.array([0, 0.5, 0.5, 0.5, 1.0])
for t in range(11):
    if t in (0, 1, 3, 10):
        print(f"t = {t:2d}           :", g[1:4])
    g = T @ g
    g[L] = fL

# Label spreading, Eqs. 34.3-9 to 34.3-11, alpha = 0.5
alpha = 0.5
S = W / np.sqrt(np.outer(d, d))
Y = np.zeros((5, 2))
Y[0, 0] = Y[4, 1] = 1
F1 = alpha * S @ Y + (1 - alpha) * Y    # one step from F = Y
Fs = (1 - alpha) * np.linalg.solve(np.eye(5) - alpha * S, Y)
print("S12 S23 S34 S45  :", np.diag(S, 1))
print("F after one step :\n", F1)
print("F* (Eq. 34.3-10) :\n", Fs)
print("P(class 1)       :", (Fs / Fs.sum(1, keepdims=True))[:, 1])

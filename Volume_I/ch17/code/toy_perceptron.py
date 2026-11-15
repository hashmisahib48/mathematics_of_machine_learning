# The perceptron on the four tiles of Section 17.1 (Example 17.7-1).
import numpy as np

X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]])  # ring, gloss
y = np.array([1, -1, 1, -1])                  # +1 sound, -1 cracked
w, b = np.zeros(2, dtype=int), 0

print("epoch  i   z  action   w        b")
for epoch in (1, 2, 3):
    mistakes = 0
    for i in range(4):
        z = y[i] * (X[i] @ w + b)      # margin, Eq. 17.3-2
        if z <= 0:                     # a mistake (zero counts)
            w = w + y[i] * X[i]        # Eq. 17.3-3
            b = b + y[i]
            mistakes += 1
        act = "update" if z <= 0 else "keep"
        print(f"{epoch:5d}  {i + 1}  {z:+3d}  {act:6s}  "
              f"{str(w.tolist()):8s} {b:+d}")
    if mistakes == 0:                  # a clean epoch: stop
        break
print("final: w =", w.tolist(), " b =", b)

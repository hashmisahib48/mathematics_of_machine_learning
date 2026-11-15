# Five finches A-E: merge sequences from the definitions.
import numpy as np

X = np.array([[3, 3], [1, 2], [0, 2], [7, 4], [11, 2]], float)
D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))

def merges(link):                        # naive agglomeration
    cl, out = [[i] for i in range(len(X))], []
    while len(cl) > 1:
        d, a, b = min((link(D[np.ix_(cl[a], cl[b])]), a, b)
                      for a in range(len(cl))
                      for b in range(a + 1, len(cl)))   # Eq. 37.3-1
        new = sorted(cl[a] + cl[b])
        out.append(f"{d:.3f} " + "".join("ABCDE"[i] for i in new))
        cl = [c for k, c in enumerate(cl)
              if k not in (a, b)] + [new]
    return " | ".join(out)

for name, f in (("single", np.min), ("complete", np.max),
                ("average", np.mean)):
    print(f"{name:9s}", merges(f))

def ss(P):                               # within-cluster SS
    return ((P - P.mean(axis=0)) ** 2).sum()

A, BC, ABC = X[[0]], X[[1, 2]], X[[0, 1, 2]]
delta = ss(ABC) - ss(A) - ss(BC)         # Eq. 37.3-3
formula = 2 / 3 * ((A[0] - BC.mean(0)) ** 2).sum()  # Eq. 37.3-4
print("Ward cost of A + BC:", round(delta, 4), round(formula, 4),
      " height", round(np.sqrt(2 * delta), 4))

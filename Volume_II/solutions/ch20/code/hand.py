# Hand checks for Problems 20A.1 - 20A.3 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 20A.1  x = 1,2,4,5,7,8; classes 0,0,1,0,1,1; query 4.6")
x = np.array([1, 2, 4, 5, 7, 8.]); y = np.array([0, 0, 1, 0, 1, 1]); q = 4.6
d = np.abs(x - q); o = np.argsort(d, kind="stable")
print("sorted points:", x[o], " distances:", d[o], " classes:", y[o], " 1/d:", 1 / d[o])
for k in range(1, 6):
    v = np.bincount(y[o[:k]], minlength=2)
    print("k =", k, " votes (class 0, class 1) =", v, " class:", int(v.argmax()),
          " tie" if v[0] == v[1] else "")
w = 1 / d[o[:3]]
print("k=3 weights:", w, " score class 0:", w[y[o[:3]] == 0].sum(), " class 1:", w[y[o[:3]] == 1].sum(),
      " share class 0:", w[y[o[:3]] == 0].sum() / w.sum())

print("\n== 20A.2  query origin, A=(3,3), B=(4,0)")
A = np.array([3, 3.]); B = np.array([4, 0.])
for q_ in (1, 2):
    print("q =", q_, " A:", (np.abs(A) ** q_).sum() ** (1 / q_), " B:", (np.abs(B) ** q_).sum() ** (1 / q_))
print("q = inf  A:", np.abs(A).max(), " B:", np.abs(B).max())
print("q=2 squared distances: A", (A ** 2).sum(), " B", (B ** 2).sum())
for q_ in (4, 8, 16, 32):
    print("q =", q_, " A: %.4f  B: %.4f" % ((np.abs(A) ** q_).sum() ** (1 / q_), (np.abs(B) ** q_).sum() ** (1 / q_)))
# the q at which A and B are equally far: 2*3^q = 4^q
print("equal distance at q =", np.log(2) / np.log(4 / 3))

print("\n== 20A.3  x = 1,2,4,7; y = 2,3,7,9; query 3.5")
x = np.array([1, 2, 4, 7.]); y = np.array([2, 3, 7, 9.]); q = 3.5
d = np.abs(x - q); o = np.argsort(d, kind="stable")
print("sorted x:", x[o], " d:", d[o], " y:", y[o], " 1/d:", 1 / d[o])
for k in (2, 3):
    nn = o[:k]; w = 1 / d[nn]
    print("k =", k, " uniform:", y[nn].mean(), " weights", w, " sum w", w.sum(), " sum w*y", (w * y[nn]).sum(), " weighted:", (w * y[nn]).sum() / w.sum())
print("k=4 uniform:", y.mean(), " (k = n gives the overall mean)")

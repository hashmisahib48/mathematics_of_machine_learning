# Six points, three classes: the hand calculations of Section 33.5.
import numpy as np

x = np.array([1., 2, 4, 5, 7, 8])
y = np.array([0, 0, 1, 1, 2, 2])

def line(xs, zs):                      # Eq. 9.3-5 of Volume I
    w = ((xs - xs.mean()) * (zs - zs.mean())).sum() \
        / ((xs - xs.mean()) ** 2).sum()
    return w, zs.mean() - w * xs.mean()

# one-versus-rest: indicator targets, three lines, Eq. 33.3-2
ovr = [line(x, (y == k) * 1.0) for k in range(3)]
S = np.array([[w * t + b for w, b in ovr] for t in x])
print("OvR lines (w, b):",
      [(round(float(w), 4), round(float(b), 4)) for w, b in ovr])
print("  x   s0      s1      s2    argmax")
for t, s in zip(x, S):
    print(f"{t:3.0f}" + "".join(f"{v:8.4f}" for v in s)
          + f"{s.argmax():5d}")

# one-versus-one: +1 for the second class of the pair, Eq. 33.3-4
f, votes = {}, np.zeros((6, 3))
for j, k in [(0, 1), (1, 2), (0, 2)]:
    m = (y == j) | (y == k)
    w, b = line(x[m], np.where(y[m] == k, 1.0, -1.0))
    f[j, k] = w * x + b
    print(f"pair ({j},{k}): f = {w:.4f} x {b:+.4f}")
    votes[np.arange(6), np.where(f[j, k] > 0, k, j)] += 1
c = np.zeros((6, 3))                   # Eq. 33.3-6
for (j, k), fp in f.items():
    c[:, j] -= fp
    c[:, k] += fp
F = votes + c / (3 * (abs(c) + 1))     # Eq. 33.3-7
print("  x     f01     f12     f02   votes      c0      c1"
      "      c2  argmax")
for i, t in enumerate(x):
    fi = [f[0, 1][i], f[1, 2][i], f[0, 2][i]]
    print(f"{t:3.0f}" + "".join(f"{v:8.3f}" for v in fi)
          + f"  {votes[i].astype(int)}"
          + "".join(f"{v:8.3f}" for v in c[i])
          + f"{F[i].argmax():6d}")
print("F at x = 4:", F[2].round(4))

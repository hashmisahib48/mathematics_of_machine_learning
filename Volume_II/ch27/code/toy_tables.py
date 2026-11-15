# Every number of the worked examples of section 27.7 (not listed in the book).
import numpy as np
from scipy.special import expit

x = np.array([1, 2, 3, 4, 5, 6.])
y = np.array([3, 5, 4, 10, 14, 12.])
eta = 0.5
F = np.full(6, y.mean())
print("F0", F[0], "MSE", ((y - F) ** 2).mean())
for m in (1, 2):
    r = y - F
    print(f"\nround {m}: r = {r}, SSE of r = {(r ** 2).sum()}")
    best = None
    for s in (x[:-1] + x[1:]) / 2:
        a, b = r[x <= s], r[x > s]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        print(f"  split {s}: left mean {a.mean():.4f} right mean "
              f"{b.mean():.4f} SSE {sse:.4f}")
        if best is None or sse < best[0]:
            best = (sse, s, a.mean(), b.mean())
    _, s, cl, cr = best
    h = np.where(x <= s, cl, cr)
    F = F + eta * h
    print("  h =", h, " F =", F, " MSE", ((y - F) ** 2).mean().round(4))
    print("  residual sums in the leaves after the step:",
          (y - F)[x <= s].sum(), (y - F)[x > s].sum())
print("\nprediction at 3.8 after 0, 1, 2 trees: 8, 10, 9.25")

print("\nabsolute error")
F0 = np.median(y)
r = y - F0
print("F0", F0, "r", r, "left", np.median(r[:3]), "right", np.median(r[3:]))
r2 = r.copy()
r2[4] = 114 - F0
print("with y5 = 114: right median", np.median(r2[3:]),
      "right mean", r2[3:].mean().round(4), " (before:", r[3:].mean(), ")")
inc = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75.])
z = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1.])
left = inc <= 32
print("\nloans: left n", left.sum(), "repaid", z[left].sum(),
      " right n", (~left).sum(), "repaid", z[~left].sum())
print("plain mean of r in the leaves:", (z - 0.5)[left].mean(),
      (z - 0.5)[~left].mean())
print("sum r over all samples:", (z - 0.5).sum())
for g in (-2.0, 4 / 3):
    print("gamma", round(g, 4), "-> F1 =", 0.5 * g, " p =",
          expit(0.5 * g).round(4))

# sum of absolute errors before and after the absolute-error round
F0 = np.full(6, 7.5)
F1 = F0 + 0.5 * np.where(np.arange(1, 7) <= 3, -3.5, 4.5)
yy = np.array([3, 5, 4, 10, 14, 12.])
print("sum |e| before", np.abs(yy - F0).sum(), "after", np.abs(yy - F1).sum())

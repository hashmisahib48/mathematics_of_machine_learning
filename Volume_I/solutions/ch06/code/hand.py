# Hand checks for the Chapter 6 class A solutions (6A.1 - 6A.4). Output: verify/hand.out
import numpy as np

print("== 6A.1  model y-hat = 1 + x")
xt = np.array([0., 1, 2, 3]); yt = np.array([1., 2, 2, 4]); xs = np.array([4., 5, 6]); ys = np.array([6., 5, 8])
for name, x, y in (("train", xt, yt), ("test", xs, ys)):
    yh = 1 + x; r = y - yh
    print(f" {name}: x {x.tolist()} y {y.tolist()} yhat {yh.tolist()} r {r.tolist()} r^2 {(r**2).tolist()} sum {np.sum(r**2):.0f} mean {np.mean(r**2):.4f}")
print(" gap =", np.mean((ys - 1 - xs) ** 2) - np.mean((yt - 1 - xt) ** 2))

print("\n== 6A.2  bias and variance, f = 5, sigma^2 = 0.5")
f, s2 = 5.0, 0.5
for name, p in (("flexible", np.array([2., 4, 4, 6])), ("stiff", np.array([3.5, 4, 4, 4.5]))):
    m = p.mean(); b2 = (m - f) ** 2; v = np.mean((p - m) ** 2); direct = s2 + np.mean((f - p) ** 2)
    print(f" {name:8s} mean {m:.3f} bias {m - f:+.3f} bias^2 {b2:.3f} var {v:.4f} -> {s2 + b2 + v:.4f};"
          f"  direct: (f-fhat)^2 = {(f - p).tolist()} -> {np.mean((f - p) ** 2):.4f}, + sigma^2 = {direct:.4f}")

print("\n== 6A.3  size of the gap, p = 6, n = 30, sigma^2 = 0.09")
p, n, s2 = 6, 30, 0.09
print(f" p/n = {p/n:.3f}  train {s2*(1-p/n):.4f}  test {s2*(1+p/n):.4f}  gap {s2*(1+p/n) - s2*(1-p/n):.4f} = 2 p s2/n = {2*p*s2/n:.4f}")
print(f" n for gap 0.006: {2*p*s2/0.006:.0f};  check at n = 180: gap {2*p*s2/180:.4f}, train {s2*(1-p/180):.4f}, test {s2*(1+p/180):.4f}")

print("\n== 6A.4  ridge, one centred feature")
x = np.array([-2., -1, 1, 2]); y = np.array([-3., -1, 1, 3]); Sxy = x @ y; Sxx = x @ x
print(f" Sxy = {Sxy:.0f}, Sxx = {Sxx:.0f}")
for a in (0, 4, 10, 18):
    w = Sxy / (Sxx + a); r = y - w * x
    print(f" alpha {a:2d}: slope {w:.4f}  shrink factor {1/(1+a/Sxx):.4f}  residuals {r.round(3).tolist()}  train MSE {np.mean(r**2):.4f}")
print(" alpha for slope 0.5:", Sxy / 0.5 - Sxx)

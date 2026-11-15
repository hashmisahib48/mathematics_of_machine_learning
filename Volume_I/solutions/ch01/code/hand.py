# Hand checks for Problems 1A.1 - 1A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)

print("== 1A.1  the table of four samples")
X = np.array([[2, 5, 1], [3, 4, 0], [7, 1, 2], [4, 6, 3]])
y = np.array([10, 9, 20, 18])
n, p = X.shape
print("n =", n, " p =", p, " shape of X =", X.shape, " shape of y =", y.shape)
print("x_23 (row 2, column 3) =", X[1, 2], "  [Python: X[1, 2]]")
print("x_3 (row 3) =", X[2])
print("feature 2 =", X[:, 1], " sum =", X[:, 1].sum(), " mean =", X[:, 1].mean())

print("\n== 1A.2  1, 2, 3, 4, 5, 6, 28")
v = np.array([1, 2, 3, 4, 5, 6, 28.])
m, md = v.mean(), np.median(v)
print("sum =", v.sum(), " mean =", m, " median =", md)
print("deviations from mean:", v - m, " squares:", (v - m) ** 2, " sum =", ((v - m) ** 2).sum())
print("MSE at the mean =", ((v - m) ** 2).mean(), " = variance", v.var())
print("absolute deviations from median:", np.abs(v - md), " sum =", np.abs(v - md).sum(),
      " MAE =", np.abs(v - md).mean(), " 33/7 =", 33 / 7)
print("for comparison: MAE at the mean =", np.abs(v - m).mean(), " MSE at the median =", ((v - md) ** 2).mean())
print("squared deviations at median:", (v - md) ** 2, " sum =", ((v - md) ** 2).sum())
print("check Eq. 1.4-6: MSE(c) = s^2 + (mean - c)^2 at c = 4:", v.var() + (m - 4) ** 2)

print("\n== 1A.3  targets 1, 4, 10; eta = 0.1")
t = np.array([1., 4, 10]); ybar = t.mean(); eta = 0.1; c = 0.0
print("mean =", ybar, " variance of the targets =", t.var())
path = [c]
for k in range(1, 4):
    g = -2 * (ybar - c)                      # Eq. 1.5-2
    c_new = c - eta * g                       # Eq. 1.5-1
    print(f"t={k-1}: c={c:.4f} slope={g:.4f} -> c_{k}={c_new:.4f}  error c-ybar={c_new-ybar:.4f}  J={np.mean((t-c_new)**2):.4f}")
    c = c_new; path.append(c)
print("closed form 5(1-0.8^t):", [round(5 * (1 - 0.8 ** k), 4) for k in range(4)])
print("factor 1-2*eta =", 1 - 2 * eta)
print("c_t after 20 steps:", 5 * (1 - 0.8 ** 20), " J(c_0) =", np.mean((t - 0) ** 2))

print("\n== 1A.4  line y = 1 + 1.5 x on the delivery data")
x = np.arange(1., 6); y = np.array([3., 4, 6, 7, 10]); yh = 1 + 1.5 * x; r = y - yh
print("prediction:", yh, " residuals:", r, " squares:", r ** 2, " abs:", np.abs(r))
print("sum of squares =", (r ** 2).sum(), " MSE =", (r ** 2).mean(), " MAE =", np.abs(r).mean())
print("for comparison, best line (Chapter 9) 0.9 + 1.7 x: SSE =", ((y - (0.9 + 1.7 * x)) ** 2).sum())
print("residual sum (not zero for this line):", r.sum())

print("\n== common slips (instructor notes)")
print("1A.1: x_32 (indices swapped) =", X[2, 1], " vs x_23 =", X[1, 2])
c = 0.0; print("1A.3: forgetting the factor 2 in the slope gives c_1 =", c - 0.1 * (-(5.0 - c)))

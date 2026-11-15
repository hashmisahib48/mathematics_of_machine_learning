# Hand checks for the Chapter 30 solutions: Problems 30A.1, 30A.2, 30A.3.
import numpy as np
np.set_printoptions(precision=4, suppress=True)
k = lambda a, b, l=1.0: np.exp(-(a - b) ** 2 / (2 * l ** 2))

print("== 30A.1")
x = np.array([0.0, 1, 3])
K = k(x[:, None], x[None, :])
print("K (ell=1) =\n", K)
print("k(0,1) = exp(-1/2)  =", round(np.exp(-0.5), 5))
print("k(1,3) = exp(-4/2)  =", round(np.exp(-2.0), 5))
print("k(0,3) = exp(-9/2)  =", round(np.exp(-4.5), 5), "   correlation =", round(K[0, 2] / np.sqrt(K[0, 0] * K[2, 2]), 5))
print("k(0,3), ell=2: exp(-9/8) =", round(np.exp(-9 / 8), 5))
print("eigenvalues of K:", np.linalg.eigvalsh(K).round(4), " all positive:", bool((np.linalg.eigvalsh(K) > 0).all()))

print("\n== 30A.2")
rho, s2, y1 = 0.8, 0.25, 1.5
print("(a) mean", rho * y1, " var", round(1 - rho ** 2, 4))
S = 1 + s2
print("(b) S =", S, " k*/S =", rho / S, " mean", round(rho / S * y1, 4), " var", round(1 - rho ** 2 / S, 4))
print("    shrink factor of the mean 0.96/1.2 =", round(0.96 / 1.2, 4), "  variance reduction 0.64 -> ", round(rho ** 2 / S, 4))
for s2 in (0.0, 0.25, 1.0, 10.0, 1e3, 1e6):
    S = 1 + s2
    print(f"(c) noise {s2:>9g}: mean {rho / S * y1:.4f}  var {1 - rho ** 2 / S:.4f}")

print("\n== 30A.3")
sn, y0 = 0.25, 1.5
S = 1 + sn
print("S = 1 + 0.25 =", S, "  a = y/S =", y0 / S)
for xs in (0.0, 1.0, 3.0):
    ks = np.exp(-xs ** 2 / 2)
    print(f"x*={xs}: k*={ks:.5f}  mean = k* a = {ks * y0 / S:.4f}  k*^2/S = {ks ** 2 / S:.5f}  var = {1 - ks ** 2 / S:.4f}")
print("noise factor S/k(0,0) =", S)
print("mean at 0 / reading =", 1 / S, "  variance at 0 =", 1 - 1 / S)

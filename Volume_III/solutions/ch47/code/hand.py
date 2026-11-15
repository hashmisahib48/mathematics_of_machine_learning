# Hand checks for the Chapter 47 solutions (Problems 47A.1 - 47A.3).
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

print("== 47A.1  data 1,2,4,9  x = 3  h = 2.5")
d = np.array([1., 2., 4., 9.]); h = 2.5; x0 = 3.0; n = len(d)
u = np.abs(x0 - d) / h
print("u =", u.round(2).tolist())
top_k = np.where(u < 1, 0.5, 0.0); epa_k = np.where(u < 1, 0.75 * (1 - u ** 2), 0.0)
print("tophat K(u)      =", top_k.round(4).tolist(), " sum", top_k.sum().round(4))
print("epanechnikov K(u) =", epa_k.round(4).tolist(), " sum", epa_k.sum().round(4))
print("n*h =", n * h)
print("tophat f(3)       =", round(top_k.sum() / (n * h), 4))
print("epanechnikov f(3) =", round(epa_k.sum() / (n * h), 4))
for k in ["tophat", "epanechnikov"]:
    f = np.exp(KernelDensity(bandwidth=h, kernel=k).fit(d[:, None]).score_samples([[x0]])[0])
    print("sklearn", k, round(float(f), 4))
print("1 - u^2 =", (1 - u ** 2).round(2).tolist(), " 0.75*(1-u^2) =", (0.75 * (1 - u ** 2)).round(3).tolist())
print("ln f tophat", round(float(np.log(top_k.sum() / (n * h))), 4), " ln f epan", round(float(np.log(epa_k.sum() / (n * h))), 4))
print("ratio epan/tophat", round(float(epa_k.sum() / top_k.sum()), 3))

print("\n== 47A.2  Scott and Silverman, n = 100")
n = 100
for p in (1, 2, 3):
    sc = n ** (-1 / (p + 4)); fac = (n * (p + 2) / 4) ** (-1 / (p + 4))
    print(f"p={p}: n^(-1/(p+4)) exponent {-1/(p+4):.4f}  Scott {sc:.4f}  (p+2)/4 = {(p+2)/4:.2f}  n(p+2)/4 = {n*(p+2)/4:.1f}  Silverman {fac:.4f}  ratio {fac/sc:.4f}")
print("ln 100 =", round(np.log(100), 4))
for p in (1, 2, 3): print(f"p={p}: exp(-ln100/(p+4)) = {np.exp(-np.log(100)/(p+4)):.4f}")
print("(4/(p+2))^(1/(p+4)):", [round((4 / (p + 2)) ** (1 / (p + 4)), 4) for p in (1, 2, 3)])
print("Silverman by Eq. 47.4-4 (4/(p+2))^(1/(p+4)) n^(-1/(p+4)):", [round((4 / (p + 2)) ** (1 / (p + 4)) * n ** (-1 / (p + 4)), 4) for p in (1, 2, 3)])
from sklearn.neighbors import KernelDensity as KD
X1 = np.random.default_rng(0).normal(size=(100, 3))
for p in (1, 2, 3):
    print("sklearn bandwidth_", p, round(KD(bandwidth="scott").fit(X1[:, :p]).bandwidth_, 4), round(KD(bandwidth="silverman").fit(X1[:, :p]).bandwidth_, 4))

print("\n== 47A.3  leave one out, waits 52 56 80 84, h = 4")
t = np.array([52., 56., 80., 84.]); h = 4.0
phi = lambda v: np.exp(-v ** 2 / 2) / np.sqrt(2 * np.pi)
print("phi(0), phi(1), phi(7), phi(8) =", [float(f"{phi(v):.6g}") for v in (0, 1, 7, 8)])
print("leave out 52:  phi(1)+phi(7)+phi(8) =", round(float(phi(1) + phi(7) + phi(8)), 6), " /12 =", round(float((phi(1) + phi(7) + phi(8)) / 12), 5))
loo = []; tr = []
for i in range(4):
    s_loo = sum(phi((t[i] - t[j]) / h) for j in range(4) if j != i)
    s_tr = sum(phi((t[i] - t[j]) / h) for j in range(4))
    f_loo = s_loo / (3 * h); f_tr = s_tr / (4 * h)
    loo.append(np.log(f_loo)); tr.append(np.log(f_tr))
    print(f"i={i+1} x={t[i]:.0f}: sum_loo {s_loo:.6f} f_loo {f_loo:.5f} ln {np.log(f_loo):.4f} | sum_train {s_tr:.6f} f {f_tr:.5f} ln {np.log(f_tr):.4f}")
print("mean LOO", round(float(np.mean(loo)), 4), " mean train", round(float(np.mean(tr)), 4), " difference", round(float(np.mean(tr) - np.mean(loo)), 4))
# identity of Section 47.4
K0 = phi(0) / h
f4 = np.exp(KernelDensity(bandwidth=h).fit(t[:, None]).score_samples(t[:, None]))
idt = np.log((4 * f4 - K0) / 3)
print("identity (n f - K(0)/h)/(n-1): ln =", idt.round(4).tolist(), " K(0)/h =", round(float(K0), 5))
print("n f(52) =", round(float(4 * f4[0]), 5), " minus K(0)/h =", round(float(4 * f4[0] - K0), 5), " /3 =", round(float((4 * f4[0] - K0) / 3), 5))
print("ln(0.75*(phi0+phi1)/phi1) =", round(float(np.log(0.75 * (phi(0) + phi(1)) / phi(1))), 4))

print("\n== 47A.2 intermediate numbers (n = 100)")
for p in (1, 2, 3):
    m = 100 * (p + 2) / 4
    print(f"p={p}: ln100/(p+4) = {np.log(100)/(p+4):.4f} -> Scott {np.exp(-np.log(100)/(p+4)):.4f};  n(p+2)/4 = {m:.0f}, ln = {np.log(m):.4f}, /(p+4) = {np.log(m)/(p+4):.4f} -> Silverman {np.exp(-np.log(m)/(p+4)):.4f}")

print("\n== 47B.2 bandwidths per fold converted to centimetres (from s47b2.out)")
sd = np.array([1.759, 0.760])
for hz in (0.10, 0.11, 0.13):
    print(f"h_z = {hz}: length {hz*sd[0]:.3f} cm, width {hz*sd[1]:.3f} cm")
for hr in (0.13, 0.16): print(f"raw h = {hr} cm in both directions; in scaled units length {hr/sd[0]:.3f}, width {hr/sd[1]:.3f}")
print("rounding step of the iris columns: 0.1 cm")
print("\n== 47A.3 ratio of the training and the leave-one-out density at 52")
print("f(52)/f_-1(52) =", round(float(((phi(0) + phi(1)) / 16) / (phi(1) / 12)), 4), " ln =", round(float(np.log(((phi(0) + phi(1)) / 16) / (phi(1) / 12))), 4))
print("(phi(0)+phi(1))/16 =", round(float((phi(0) + phi(1)) / 16), 5), " phi(1)/12 =", round(float(phi(1) / 12), 5))
print("phi(0) =", round(float(phi(0)), 5), " phi(0)/4 =", round(float(phi(0) / 4), 5), " 4 f(52) =", round(float(4 * (phi(0) + phi(1)) / 16), 5))
print("\n== slips quoted in the instructor notes")
print("47A.3 divide the leave-one-out sum by n h = 16 instead of (n-1) h = 12: f =", round(float(phi(1) / 16), 5), " ln =", round(float(np.log(phi(1) / 16)), 4))
print("47A.3 second point (56): phi(1)+phi(6)+phi(7) =", float(f"{phi(1) + phi(6) + phi(7):.9g}"), " phi(6) =", float(f"{phi(6):.3g}"))
print("47B.2 geometric mean of 0.193 and 0.084 cm:", round(float(np.sqrt(0.193 * 0.084)), 3))
print("47B.2 wrong sign of the Jacobian: -1.138 - 0.290 =", round(-1.138 - 0.290, 3), " correct: -1.138 + 0.290 =", round(-1.138 + 0.290, 3))
print("47A.1 forgetting to divide by h:", 1.5 / 4, " and counting the point 9 in the Epanechnikov sum: 1.53 + 0.75*(1-2.4**2) =", round(1.53 + 0.75 * (1 - 2.4 ** 2), 2))

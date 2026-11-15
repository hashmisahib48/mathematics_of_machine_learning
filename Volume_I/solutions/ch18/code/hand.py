# Hand checks for Problems 18A.1 - 18A.4 (every number quoted in the solutions)
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sig = lambda t: 1 / (1 + np.exp(-t))

print("== 18A.1  means 2 and 6, variance 2, priors 0.7 and 0.3")
m0, m1, v, p0, p1 = 2.0, 6.0, 2.0, 0.7, 0.3
w = (m1 - m0) / v; b = -(m1 ** 2 - m0 ** 2) / (2 * v) + np.log(p1 / p0)
print("w =", w, " b = -(36-4)/(2*2) + ln(0.3/0.7) =", round(-(m1 ** 2 - m0 ** 2) / (2 * v), 4), "+", round(np.log(p1 / p0), 4), "=", round(b, 4))
xs = (m0 + m1) / 2 + v / (m1 - m0) * np.log(p0 / p1)
print("x* = 4 + (2/4) ln(0.7/0.3) = 4 +", round(0.5 * np.log(p0 / p1), 4), "=", round(xs, 4), "  shift from the midpoint:", round(xs - 4, 4), "  -b/w =", round(-b / w, 4))
f = w * 4.5 + b; print("log-odds at 4.5:", round(f, 4), " P(class 1) =", round(sig(f), 4), " P(class 0) =", round(1 - sig(f), 4))
print("equal priors: b =", -(m1 ** 2 - m0 ** 2) / (2 * v), " log-odds at 4.5:", w * 4.5 - 8, " P =", round(sig(w * 4.5 - 8), 4))
print("log-odds at 4.0:", w * 4 + b, " P(class 1 | 4) =", round(sig(w * 4 + b), 4))
print("wrong sign of the logarithm gives the boundary", round(4 - 0.5 * np.log(p0 / p1), 4))
print("shift sigma^2/(mu1-mu0) ln(pi0/pi1) for means 2 and 6:", round(xs - 4, 4), "; for means 2 and 12:", round(2 / 10 * np.log(7 / 3), 4), "; for variance 0.5:", round(0.5 / 4 * np.log(7 / 3), 4))

print("\n== 18A.2  Sigma = diag(2, 0.5), means (0,0) and (4,2), x = (2.5, 0.5)")
S = np.diag([2.0, 0.5]); Si = np.linalg.inv(S); mu0 = np.array([0., 0.]); mu1 = np.array([4., 2.]); x = np.array([2.5, 0.5])
for k, m in enumerate((mu0, mu1)):
    d = x - m; print(f"class {k}: x - mu = {d}  squares {d ** 2}  Euclid = {np.linalg.norm(d):.4f}  (x-mu)^2/Sigma = {d ** 2 / np.diag(S)}  d_M^2 = {d @ Si @ d:.4f}")
wv = Si @ (mu1 - mu0); bb = -0.5 * (mu0 + mu1) @ wv; sc = wv @ x + bb
print("Sigma^-1 =", np.diag(Si), " w =", wv, " b =", bb, " score =", sc, " P(1) =", round(sig(sc), 4))
print("check: 0.5*(d_M0^2 - d_M1^2) =", 0.5 * ((x - mu0) @ Si @ (x - mu0) - (x - mu1) @ Si @ (x - mu1)))
print("boundary through the midpoint (2, 1):", wv @ np.array([2., 1.]) + bb)

print("\n== 18A.3  parameter counts")
cl = lambda K, p: K * p + p * (p + 1) // 2 + K - 1
cq = lambda K, p: K * p + K * (p * (p + 1) // 2) + K - 1
for K, p, nk in ((4, 20, 75), (4, 80, 75)):
    print(f"K={K} p={p} n_k={nk} n={K * nk}: means {K * p}, pooled covariance {p * (p + 1) // 2}, priors {K - 1} -> LDA {cl(K, p)};  4 covariances {K * (p * (p + 1) // 2)} -> QDA {cq(K, p)}")
    print(f"   rank of a class covariance <= n_k - 1 = {nk - 1} against p = {p}: {'singular' if nk - 1 < p else 'can be full'};  rank of the pooled covariance <= n - K = {K * nk - K}: {'singular' if K * nk - K < p else 'can be full'}")
    print(f"   parameters per training sample: LDA {cl(K, p) / (K * nk):.2f}  QDA {cq(K, p) / (K * nk):.2f}")

print("\n== 18A.4  Fisher's criterion along chosen directions (Example 18.6-3)")
X8 = np.array([[1, 1], [3, 3], [1, 2], [3, 2], [4, 2], [6, 4], [5, 2], [5, 4]], float); y8 = np.r_[0, 0, 0, 0, 1, 1, 1, 1]
D0 = X8[y8 == 0] - X8[y8 == 0].mean(0); D1 = X8[y8 == 1] - X8[y8 == 1].mean(0)
Sig = (D0.T @ D0 + D1.T @ D1) / 8; d = X8[y8 == 1].mean(0) - X8[y8 == 0].mean(0)
print("Sigma =", Sig.tolist(), " d =", d)
J = lambda a: (a @ d) ** 2 / (a @ Sig @ a)
for name, a in (("(1,0)", np.array([1., 0])), ("(0,1)", np.array([0., 1])), ("(3,1) = d", d), ("(7,-3)", np.array([7., -3]))):
    print(f"a = {name:9s}  a.d = {a @ d:5.1f}  a'Sigma a = {a @ Sig @ a:7.4f}  J = {J(a):.4f}")
print("d' Sigma^-1 d =", round(d @ np.linalg.solve(Sig, d), 4))
r = Sig[0, 1] / np.sqrt(Sig[0, 0] * Sig[1, 1]); print("within-class correlation:", round(r, 4))
print("best direction scan: max J over 3600 angles =", round(max(J(np.array([np.cos(t), np.sin(t)])) for t in np.linspace(0, np.pi, 3600, endpoint=False)), 4))

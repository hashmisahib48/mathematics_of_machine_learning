# Hand checks for the Chapter 22 solutions (22A.1 - 22A.4); every number quoted in the text comes from here.
import numpy as np
print("== 22A.1")
w = np.array([3., 4.]); b = -10.; P = np.array([4., 3.]); nw = np.linalg.norm(w)
f = w @ P + b
print("||w|| =", nw, " f(P) =", f, " d(P) =", f / nw, " f(0) =", b, " d(0) =", b / nw, " gap 2/||w|| =", 2 / nw)
foot = P - f / nw**2 * w
print("foot of the perpendicular", foot.round(3), " f(foot) =", round(w @ foot + b, 12))
print("== 22A.2")
x1, x2 = np.array([1., 2.]), np.array([3., 2.])
alpha = 2 / ((x2 - x1) @ (x2 - x1))
wv = alpha * (x2 - x1); bv = 1 - wv @ x2
print("alpha =", alpha, " w =", wv, " b =", bv, " width =", 2 / np.linalg.norm(wv))
print("f(x1), f(x2) =", wv @ x1 + bv, wv @ x2 + bv, " |w|^2 =", wv @ wv, " sum alpha =", 2 * alpha, " W =", 2 * alpha - 0.5 * wv @ wv)
print("== 22A.3")
yf = np.array([1.8, 1.0, 0.4, -0.5]); xi = np.maximum(0, 1 - yf)
print("xi =", xi.round(2), " sum =", xi.sum().round(2), " C*sum =", (2 * xi.sum()).round(2))
print("== 22A.4")
Kii, Kjj, Kij, yi, yj, ai, aj, Ei, Ej, C = 5, 4, 2, 1, -1, .1, .1, -.4, .3, 1.
eta = Kii + Kjj - 2 * Kij
ajn = aj + yj * (Ei - Ej) / eta
L, H = max(0, aj - ai), min(C, C + aj - ai)
ajc = min(max(ajn, L), H); ain = ai + yi * yj * (aj - ajc)
print("eta =", eta, " E_i - E_j =", Ei - Ej, " aj_new =", round(ajn, 4), " L, H =", L, H, " aj clipped =", round(ajc, 4))
print("ai_new =", round(ain, 4), " sum a y before =", yi * ai + yj * aj, " after =", round(yi * ain + yj * ajc, 12))
dW = yj * (Ei - Ej) * (ajc - aj) - 0.5 * eta * (ajc - aj)**2     # gain of W along the line, Eq. 22.6-2
print("slope of W along the line at a_j =", yj * (Ei - Ej), " gain of W =", round(dW, 4))

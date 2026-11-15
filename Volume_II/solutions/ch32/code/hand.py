# Hand checks for Problems 32A.1 - 32A.3 (every number quoted in the solutions)
import numpy as np
from sklearn.isotonic import isotonic_regression
sg = lambda z: 1 / (1 + np.exp(-z))

print("== 32A.1  y = 6 2 4 3 8, w = 1 1 2 1 1  (stack of blocks as value(weight))")
y = [6, 2, 4, 3, 8.]; w = [1, 1, 2, 1, 1.]; st = []
for i, (yi, wi) in enumerate(zip(y, w), 1):
    v, W = yi, wi; act = "push"
    while st and st[-1][0] > v:
        v0, W0 = st.pop(); nv = (W0 * v0 + W * v) / (W0 + W)
        act = f"merge {v0:g}({W0:g}) with {v:.4f}({W:g}) -> {nv:.4f}({W0 + W:g})"; v, W = nv, W0 + W
    st.append((v, W))
    print(i, yi, wi, act, " | stack:", " | ".join(f"{a:.4f}({b:g})" for a, b in st))
f = isotonic_regression(np.array(y), sample_weight=np.array(w))
print("fit (sklearn)      :", f.round(4))
print("sum w*y =", (np.array(w) * y).sum(), " sum w*f =", (np.array(w) * f).sum())
print("squared error sum w (y-f)^2 =", (np.array(w) * (np.array(y) - f) ** 2).sum().round(4))
print("block 1: (6*1+2*1)/2 =", 8 / 2, "; (4*2+3*1)/3 =", round(11 / 3, 4), "; (4*2+11/3*3)/5 =", (8 + 11) / 5)
print("sum of w*y of samples 1..4 =", 6 + 2 + 8 + 3, "weight 5 ->", 19 / 5)

print("\n== 32A.2  forecasts 0.8 (4 cases, 3 events), 0.2 (6 cases, 2 events)")
q = np.array([.8, .2]); n = np.array([4, 6]); h = np.array([3, 2]); N = n.sum()
ok = h / n; o = h.sum() / N
print("observed:", ok.round(4), " overall rate:", o)
rel_t = n * (q - ok) ** 2; res_t = n * (ok - o) ** 2
print("REL terms n(q-o)^2:", rel_t.round(5), " sum/N =", (rel_t.sum() / N).round(4))
print("RES terms n(o_k-o)^2:", res_t.round(5), " sum/N =", (res_t.sum() / N).round(4))
unc = o * (1 - o); print("UNC =", unc)
bs = rel_t.sum() / N - res_t.sum() / N + unc; print("BS =", round(bs, 4))
yy = np.concatenate([[1] * a + [0] * (b - a) for a, b in zip(h, n)]); qq = np.repeat(q, n)
sq = (yy - qq) ** 2
print("group sums: events at 0.8:", 3 * 0.2 ** 2, " non-events at 0.8:", 1 * 0.8 ** 2, " events at 0.2:", 2 * 0.8 ** 2, " non-events at 0.2:", 4 * 0.2 ** 2)
print("direct squared errors: events at 0.8:", 0.2 ** 2, " non-events at 0.8:", 0.8 ** 2,
      " events at 0.2:", 0.8 ** 2, " non-events at 0.2:", 0.2 ** 2)
print("sum of squared errors:", sq.sum().round(4), " mean:", sq.mean().round(4))
print("constant 0.5: mean((y-0.5)^2) =", ((yy - .5) ** 2).mean(), " (UNC =", unc, ")")
print("share of BS removed by the forecaster:", round((unc - bs) / unc, 3))

print("\n== 32A.3  scores -2 -1 1 2, labels 0 1 0 1")
f3 = np.array([-2, -1, 1, 2.]); l3 = np.array([0, 1, 0, 1]); npos = l3.sum(); nneg = len(l3) - npos
tp = (npos + 1) / (npos + 2); tn = 1 / (nneg + 2); t = np.where(l3 == 1, tp, tn)
print("n+ =", npos, " n- =", nneg, " t+ =", tp, " t- =", tn, " t =", t)
print("sum f t =", (f3 * t).sum(), " sum f^2 =", (f3 ** 2).sum(), " sum t =", t.sum())
s = 0.0
for k in range(4):
    p = sg(s * f3); g = (f3 * (p - t)).sum(); hh = (f3 ** 2 * p * (1 - p)).sum(); step = -g / hh
    print(f"step {k}: s = {s:.6f} p = {p.round(4)} g = {g:.6f} h = {hh:.5f} update = {step:+.6f}")
    s += step
print("final s =", round(s, 6), " A =", round(-s, 6), " P(f=1) =", round(sg(s), 4))
p = sg(s * f3); print("calibrated probabilities:", p.round(4), " sum(t-p) =", (t - p).sum().round(12), " sum f(t-p) =", (f3 * (t - p)).sum().round(10))
print("B check with a joint fit:")
from scipy.optimize import minimize
J = lambda ab: -(t * np.log(sg(-(ab[0] * f3 + ab[1]))) + (1 - t) * np.log(1 - sg(-(ab[0] * f3 + ab[1])))).sum()
r = minimize(J, [0, 0], method="BFGS", options={"gtol": 1e-10}); print("A, B =", r.x.round(5))

print("\n== slips quoted in the instructor notes")
yv = np.array(y); wv = np.array(w)
fu = isotonic_regression(yv)
print("32A.1 weights ignored: fit", fu.round(4), " sum w f =", (wv * fu).sum(), " (sum w y = 27)")
for nm, tt in (("labels as targets", l3.astype(float)),):
    s = 0.0
    for k in range(30):
        p = sg(s * f3); g = (f3 * (p - tt)).sum(); hh = (f3 ** 2 * p * (1 - p)).sum(); s -= g / hh
    print("32A.3", nm, ": s =", round(s, 5), " P(f=1) =", round(sg(s), 4))

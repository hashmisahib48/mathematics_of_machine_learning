# Problem 11C.2: two identical columns, lasso against elastic net. Writes fig/s11c2.png
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import Lasso, ElasticNet

dist = np.array([1., 2, 3, 4, 5]); t = np.array([3., 4, 6, 7, 10]); X = np.c_[dist, dist]
Xc, yc = X - X.mean(0), t - t.mean(); n = 5
v = (Xc ** 2).mean(0); c = Xc.T @ yc / n
print("v =", v, " c =", c, " mean product of the two columns =", Xc[:, 0] @ Xc[:, 1] / n)
las = Lasso(alpha=1.0, tol=1e-12).fit(X, t); en = ElasticNet(alpha=1.0, l1_ratio=0.5, tol=1e-12).fit(X, t)
print("Lasso(alpha=1):            w =", las.coef_.round(4), " sum =", las.coef_.sum().round(4), " b =", round(las.intercept_, 4))
print("ElasticNet(1, 0.5):        w =", en.coef_.round(4), " sum =", en.coef_.sum().round(4), " b =", round(en.intercept_, 4))
print("one column alone (11.5-1): w = (3.4 - 1)/2 =", (3.4 - 1) / 2)
# coordinate descent by hand on the lasso: after w1, the second column sees z2 = c - v w1
w1 = (c[0] - 1.0) / v[0]; z2 = c[1] - (Xc[:, 0] @ Xc[:, 1] / n) * w1
print("cyclic descent: w1 =", w1, " then z2 = c2 - 2 w1 =", z2, " = alpha, so w2 stays 0")
seeds = [Lasso(alpha=1.0, tol=1e-12, selection='random', random_state=s).fit(X, t).coef_ for s in range(20)]
first = sum(1 for w in seeds if w[0] > 0); second = sum(1 for w in seeds if w[1] > 0)
print("selection='random', seeds 0-19: all weight on column 1 in %d fits, on column 2 in %d fits; both non-zero in %d" % (first, second, sum(1 for w in seeds if (w > 0).all())))
print("sums of the two weights over the 20 seeds:", sorted(set(round(float(w.sum()), 4) for w in seeds)))
def J(w1_, w2_, lam1, lam2):
    r = yc - Xc[:, 0] * w1_ - Xc[:, 1] * w2_
    return float((r ** 2).sum() / (2 * n) + lam1 * (abs(w1_) + abs(w2_)) + 0.5 * lam2 * (w1_ ** 2 + w2_ ** 2))
S = 1.2; s = np.linspace(0, 1, 11)
print("split share s of w1 + w2 = 1.2 :", s.round(1).tolist())
print("lasso objective along the split:", [round(J(S * q, S * (1 - q), 1.0, 0.0), 4) for q in s])
S2 = 1.2889
print("enet objective along the split (sum 1.2889):", [round(J(S2 * q, S2 * (1 - q), 0.5, 0.5), 4) for q in s])
u = (c[0] - 0.5) / (2 * v[0] + 0.5)
print("elastic net by symmetry: u = (c - lambda1)/(2v + lambda2) = (3.4 - 0.5)/(4 + 0.5) =", round(u, 4), " sum 2u =", round(2 * u, 4))
# the solutions for other splits of the same sum are worse for the elastic net
q = np.linspace(0, 1, 201)
best = {}
for lam in ((1.0, 0.0), (0.5, 0.5)):
    sums = np.linspace(0.5, 2, 601); Jm = [(J(sm * a_, sm * (1 - a_), *lam), sm, a_) for sm in sums for a_ in q]; m = min(Jm)
    best[lam] = m; print("grid minimum for (lambda1, lambda2) =", lam, ": J = %.4f at sum %.3f%s" % (m[0], m[1], ", every share ties" if lam[1] == 0 else ", share %.3f" % m[2]))
print("elastic net objective at the lasso solution (1.2, 0): %.4f ; at its own solution: %.4f" % (J(1.2, 0, .5, .5), J(u, u, .5, .5)))

fig, ax = plt.subplots(figsize=(3.3, 2.0), dpi=240)
ax.plot(s, [J(S * q_, S * (1 - q_), 1.0, 0.0) for q_ in s], color=GREEN, lw=1.6, marker='o', ms=3, label='lasso, $w_1+w_2=1.2$')
ax.plot(s, [J(S2 * q_, S2 * (1 - q_), 0.5, 0.5) for q_ in s], color=OR, lw=1.6, ls='--', marker='s', ms=3, label='elastic net, $w_1+w_2=1.289$')
ax.set_xlabel('share of the sum carried by column 1', fontsize=8.5); ax.set_ylabel('objective', fontsize=8.5); ax.tick_params(labelsize=8.5)
ax.legend(fontsize=8.5, loc='upper center', borderpad=0.2, handlelength=1.8); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
lo, hi = ax.get_ylim(); ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.4 * (hi - lo))
fig.tight_layout(pad=0.4)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s11c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s11c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 3.5 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.5 / w))

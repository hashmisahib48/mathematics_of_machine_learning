# Problem 32C.2: the zero of an isotonic calibrator. Writes fig/s32c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB

print("one case with p = 0 that is an event")
eps = np.finfo(float).eps
print("  log_loss clips p at eps = %.3g: -ln(eps) = %.2f   Brier contribution (1-0)^2 = 1" % (eps, -np.log(eps)))
for lo in (0.001, 0.01, 0.05): print(f"  y_min = {lo}: -ln = {-np.log(lo):.2f}")
rng = np.random.default_rng(0)                     # the data of Section 32.4
Z = rng.normal(size=(25000, 3)); X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape) for _ in range(3)])
y = (rng.random(25000) < 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))).astype(int)
nb = GaussianNB().fit(X[:5000], y[:5000]); Xca, yca, Xte, yte = X[5000:15000], y[5000:15000], X[15000:], y[15000:]
bounds = [0, 0.001, 0.003, 0.01, 0.03, 0.1]; sizes = [50, 200, 1000]; R = {}
def run(idx, lo, meth='isotonic'):
    c = CalibratedClassifierCV(FrozenEstimator(nb), method=meth)
    if meth == 'isotonic':
        from sklearn.isotonic import IsotonicRegression
        s = nb.predict_proba(Xca[idx])[:, 1]; iso = IsotonicRegression(y_min=lo, y_max=1 - lo, out_of_bounds='clip').fit(s, yca[idx])
        p = iso.predict(nb.predict_proba(Xte)[:, 1])
    else: p = c.fit(Xca[idx], yca[idx]).predict_proba(Xte)[:, 1]
    return log_loss(yte, p), brier_score_loss(yte, p), int((p == 0).sum() + (p == 1).sum())
print("\nlog loss on 10 000 test cases, mean of 10 draws; isotonic with y_min = lo, y_max = 1 - lo (lo = 0: plain)")
print("   n   " + "".join(f"lo={b:<8g}" for b in bounds) + " sigmoid   (Brier at lo=0, 0.01, sigmoid; exact 0/1 forecasts at lo=0)")
for n in sizes:
    L = np.zeros((10, len(bounds))); Bq = np.zeros((10, len(bounds))); S = []; Sb = []; Z01 = []
    for r in range(10):
        idx = np.random.default_rng(100 * n + r).choice(10000, n, replace=False)
        for j, lo in enumerate(bounds):
            l, b, z = run(idx, lo); L[r, j] = l; Bq[r, j] = b
            if lo == 0: Z01.append(z)
        l, b, _ = run(idx, 0, 'sigmoid'); S.append(l); Sb.append(b)
    R[n] = L.mean(0); print(f"{n:5d}  " + "".join(f"{v:<11.4f}" for v in L.mean(0)) + f"{np.mean(S):.4f}   (Brier {Bq.mean(0)[0]:.4f}, {Bq.mean(0)[3]:.4f}, {np.mean(Sb):.4f}; {np.mean(Z01):.0f} exact 0/1)")
fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=240)
xs = np.arange(len(bounds))
for n, col, mk in zip(sizes, (RED, OR, BLUE), 'osv'): ax.plot(xs, R[n], '-' + mk, color=col, ms=3.5, lw=1.1, label=f'{n} calibration cases')
ax.set_xticks(xs); ax.set_xticklabels(['0'] + [f'{b:g}' for b in bounds[1:]]); ax.set_xlabel('bound $y_{min}$ (and $y_{max}$ = 1 − $y_{min}$)'); ax.set_ylabel('test log loss')
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(fontsize=7.8, loc='upper right'); ax.tick_params(labelsize=7.8); ax.xaxis.label.set_size(8.2); ax.yaxis.label.set_size(8.2)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s32c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s32c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 3.4 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.4 / w))

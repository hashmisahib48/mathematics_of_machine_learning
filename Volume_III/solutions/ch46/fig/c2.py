# Problem 46C.2: the largest distance of the empirical covariance, and MinCovDet when p is not small against n. Writes fig/s46c2.png, verify/c2.out
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch46/code'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import EmpiricalCovariance, MinCovDet

print("== the largest squared distance of a training sample under the empirical covariance (divisor n)")
cut4, cut20 = chi2(4).isf(0.025), chi2(20).isf(0.025)
print("cut-offs: chi2_4 %.3f, chi2_20 %.3f" % (cut4, cut20))
print("   n  extreme sample (n-1 equal points, one apart), p = 1:  d^2   n-1   (n-1)^2/n")
for n in (5, 13, 14, 30, 71):
    x = np.zeros((n, 1)); x[-1] = 1.0; d2 = EmpiricalCovariance().fit(x).mahalanobis(x).max()
    print("%4d %55.3f %6d %10.3f" % (n, d2, n - 1, (n - 1)**2 / n))
print("smallest n with n-1 > chi2_4 cut-off: %d;  with (n-1)^2/n > cut-off: %d" % (next(n for n in range(2, 99) if n - 1 > cut4), next(n for n in range(2, 99) if (n - 1)**2 / n > cut4)))
rng = np.random.RandomState(0); mx = max(EmpiricalCovariance().fit(Z).mahalanobis(Z).max() for Z in (rng.standard_t(2, size=(30, 20)) for _ in range(300)))
print("heavy-tailed data (t with 2 df), n = 30, p = 20, 300 draws: the largest d^2 ever seen %.2f  (bound n-1 = 29 < cut-off %.2f: no sample can be flagged)" % (mx, cut20))

print("\n== MinCovDet on clean normal data: share of samples above the 97.5 per cent point")
def share(n, p, reps, seed=0):
    r = np.random.RandomState(seed); f = []
    for _ in range(reps):
        Z = r.normal(size=(n, p))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning); m = MinCovDet(random_state=0).fit(Z)
        f.append((m.dist_ > chi2(p).isf(0.025)).mean())
    return np.mean(f)
ps = [2, 4, 6, 8, 10, 13, 16, 20]
sh60 = [share(60, p, 15) for p in ps]; sh30 = [share(30, p, 15) for p in ps]
print("   p   h (n=60)  p/h    share flagged n = 60     h (n=30)  p/h   share flagged n = 30")
for p, a, b in zip(ps, sh60, sh30): print("%4d %9d %6.2f %16.3f %16d %6.2f %16.3f" % (p, np.ceil((60 + p + 1) / 2), p / np.ceil((60 + p + 1) / 2), a, np.ceil((30 + p + 1) / 2), p / np.ceil((30 + p + 1) / 2), b))
print("nominal share 0.025")

print("\n== the wine of Problem 46B.3 with all 13 features (59 clean wines and 12 intruders, as in mcd_wine.py)")
wine = load_wine(); y = wine.target; bad = np.random.RandomState(0).choice(np.where(y == 2)[0], 12, replace=False)
W = np.vstack([wine.data[y == 0], wine.data[bad]]); isb = np.r_[np.zeros(59, bool), np.ones(12, bool)]
for nm, cols in (("4 features", [0, 6, 9, 12]), ("13 features", list(range(13)))):
    A = W[:, cols]; m = MinCovDet(random_state=0).fit(A); q = len(cols); f = m.dist_ > chi2(q).isf(0.025); h = int(np.ceil((71 + q + 1) / 2))
    e = np.linalg.eigvalsh(np.cov(A[m.raw_support_].T, bias=True)); e0 = np.linalg.eigvalsh(np.cov(A[~isb].T, bias=True))
    print("%-11s h = %d, p/h = %.2f; caught %2d of 12, false alarms %2d of 59;  raw support: %d samples, %d intruders;  eigenvalue ratio smallest/largest: raw support %.2e, the 59 clean wines %.2e" % (
        nm, h, q / h, f[isb].sum(), f[~isb].sum(), m.raw_support_.sum(), (m.raw_support_ & isb).sum(), e[0] / e[-1], e0[0] / e0[-1]))
print("Marchenko-Pastur edges (1 -+ sqrt(p/h))^2 for p = 13, h = 43: %.2f %.2f;  for p = 20, h = 26: %.3f %.2f" % ((1 - np.sqrt(13 / 43))**2, (1 + np.sqrt(13 / 43))**2, (1 - np.sqrt(20 / 26))**2, (1 + np.sqrt(20 / 26))**2))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240); ax = axs[0]
ax.plot(ps, sh60, 'o-', color=BLUE, ms=3, lw=1.1, label='$n=60$'); ax.plot(ps, sh30, 's-', color=OR, ms=3, lw=1.1, label='$n=30$'); ax.axhline(0.025, color=GREEN, ls='--', lw=1.0)
ax.text(10.6, 0.04, 'nominal 0.025', color=GREEN, fontsize=8.8); ax.set_xlabel('number of features $p$'); ax.set_ylabel('clean samples flagged'); ax.legend(loc='upper left', handlelength=1.2, labelspacing=0.2, fontsize=8.8)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax = axs[1]; nn = np.arange(2, 81); ax.plot(nn, nn - 1, color=NAVY, lw=1.3); ax.text(52, 38, 'largest $d^{2}$\n$=n-1$', color=NAVY, fontsize=8.8, ha='left')
for pp, col in ((4, BLUE), (20, OR)): ax.axhline(chi2(pp).isf(0.025), color=col, ls='--', lw=1.1); ax.text(2, chi2(pp).isf(0.025) + 2, '$\\chi^{2}_{%d}$ cut-off' % pp, color=col, fontsize=8.8)
ax.set_xlabel('number of samples $n$'); ax.set_ylabel('squared distance'); ax.set_ylim(0, 80); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4, w_pad=1.0)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s46c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s46c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

# Problem 45C.1: why n_components="auto" fails on the digits. Writes fig/s45c1.png, verify/c1.out
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim as jl

X, y = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
n, p = Xtr.shape
print("n = %d, p = %d" % (n, p))
for e in (0.5, 0.3, 0.1, 0.9, 0.99, 0.999): print("  eps %.3f: JL dimension %d  (%.1f times p)" % (e, jl(n, eps=e), jl(n, eps=e) / p))
print("minimum of 4 ln n / (eps^2/2 - eps^3/3) over 0 < eps < 1, reached at eps -> 1: %.1f" % (4 * np.log(n) / (1 / 2 - 1 / 3)))
for e in (0.5, 0.1):
    try: GaussianRandomProjection(n_components='auto', eps=e, random_state=0).fit(Xtr)
    except ValueError as err: print("auto, eps = %.1f -> ValueError: %s" % (e, str(err)[:150]))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always'); rp = GaussianRandomProjection(6176, random_state=0).fit(Xtr)
    print("explicit k = 6176 -> %s: %s" % (w[0].category.__name__, str(w[0].message)[:120]))
Z = rp.transform(Xtr); print("a 6176-column table: %.1f MB against %.2f MB for the pixels; rank of Z = %d, rank of the pixel table = %d" % (Z.nbytes / 1e6, Xtr.nbytes / 1e6, np.linalg.matrix_rank(Z), np.linalg.matrix_rank(Xtr)))
a, b = np.random.RandomState(1).randint(0, n, (2, 3000)); m = a != b; a, b = a[m], b[m]
q = ((Z[a] - Z[b])**2).sum(1) / ((Xtr[a] - Xtr[b])**2).sum(1); print("k = 6176: q has mean %.3f and sd %.3f over 3000 pairs (sqrt(2/k) = %.3f)" % (q.mean(), q.std(), np.sqrt(2 / 6176)))

ks = [2, 4, 8, 16, 32, 64, 128, 512, 6176]; acc = []
for k in ks:
    a_ = []
    for s in range(5):
        r = GaussianRandomProjection(k, random_state=s).fit(Xtr) if k <= p else None
        if r is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore'); r = GaussianRandomProjection(k, random_state=s).fit(Xtr)
        a_.append(KNeighborsClassifier(5).fit(r.transform(Xtr), ytr).score(r.transform(Xte), yte))
    acc.append((np.mean(a_), np.std(a_)))
raw = KNeighborsClassifier(5).fit(Xtr, ytr).score(Xte, yte)
print("5-NN test accuracy of the pixels: %.4f" % raw)
for k, (m_, s_) in zip(ks, acc): print("  k = %4d: %.3f +- %.3f   (%d%% of p, %.1f%% of raw accuracy)" % (k, m_, s_, round(100 * k / p), 100 * m_ / raw))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240)
es = np.linspace(0.05, 0.99, 100); ax = axs[0]
ax.plot(es, jl(n, eps=es), color=BLUE, lw=1.4); ax.axhline(p, color=RED, lw=1.2); ax.text(0.05, 72, 'number of features $p=64$', color=RED, fontsize=8.2, va='bottom')
ax.set_yscale('log'); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('$k$ required by Eq. 45.3-3'); ax.set_ylim(40, 2e5); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title('what "auto" asks for', fontsize=8.6, pad=3)
ax = axs[1]; ax.errorbar(ks, [a[0] for a in acc], yerr=[a[1] for a in acc], fmt='-o', color=BLUE, ms=3, lw=1.1, capsize=2)
ax.axhline(raw, color=GREEN, lw=1.1, ls='--'); ax.text(2.2, raw - 0.01, 'pixels', color=GREEN, fontsize=8.2, va='top'); ax.axvline(p, color=RED, lw=1.0)
ax.set_xscale('log'); ax.set_xlabel('$k$'); ax.set_ylabel('5-NN accuracy'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title('what is gained', fontsize=8.6, pad=3)
fig.tight_layout(pad=0.4, w_pad=1.0)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s45c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s45c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

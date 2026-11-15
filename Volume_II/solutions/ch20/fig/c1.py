# Problem 20C.1: best k against n. Writes fig/s20c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

ns = [50, 100, 200, 400, 800, 1600]; ks = list(range(1, 202, 2)); curves = {}; best = []; book = []; mid = []

def cv_curve(X, y, ok):               # five stratified folds as cross_val_score; one neighbour search serves every k
    acc = np.zeros(len(ok))
    for tr, te in StratifiedKFold(5).split(X, y):
        nn = KNeighborsClassifier(max(ok)).fit(X[tr], y[tr]).kneighbors(X[te], return_distance=False)
        lab = y[tr][nn]
        acc += np.array([((lab[:, :k].mean(1) > 0.5) == y[te]).mean() for k in ok])    # odd k: no ties
    return acc / 5

for n in ns:                                  # the data sets of the book's figure: 12 sets for each n
    ok = [k for k in ks if k <= 0.8 * n]
    m = np.mean([cv_curve(*make_moons(n, noise=0.3, random_state=10 + s), ok) for s in range(12)], 0)
    curves[n] = (ok, m); b61 = [k for k in ok if k <= 61]
    best.append(ok[int(np.argmax(m))]); book.append(b61[int(np.argmax(m[:len(b61)]))])
    flat = [k for k, v in zip(ok, m) if v >= m.max() - 0.003]; mid.append(float(np.sqrt(min(flat) * max(flat))))
    print("n = %4d: best k on the book's grid (k <= 61) = %2d; on the grid up to 201 = %3d (accuracy %.4f, k/n = %.4f); "
          "ks within 0.003 of the best: %d to %d" % (n, book[-1], best[-1], m.max(), best[-1] / n, min(flat), max(flat)))
print("best k, book grid:", book, " full grid:", best, " middle of the plateau:", [round(v) for v in mid])
print("middle of plateau / n:", [round(v / n, 4) for v, n in zip(mid, ns)])
lo = np.polyfit(np.log(ns), np.log(mid), 1)
print("log-log slope of the plateau middle against n: %.2f (a pure square root would be 0.50)" % lo[0])
print("bias-variance picture: variance of a k-average ~ 1/k, bias of an average over a neighbourhood of radius r ~ r^2 with r ~ sqrt(k/n) in two dimensions")

fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.0), dpi=240, gridspec_kw={'width_ratios': [1.3, 1]})
for n, c_ in ((100, PUR), (400, GREEN), (1600, OR)):
    ok, m = curves[n]; ax = axs[0]
    ax.plot(ok, m, color=c_, lw=1.4, label='n = %d' % n); ax.plot(ok[int(np.argmax(m))], m.max(), 'o', color=c_, ms=4)
axs[0].set_xlabel('k', fontsize=9); axs[0].set_xlim(0, 150); axs[0].set_ylabel('cross-validated accuracy', fontsize=8.5); axs[0].legend(fontsize=8, loc='lower right'); axs[0].set_ylim(0.8, 0.93)
axs[1].loglog(ns, mid, '-o', color=PUR, ms=3.5, lw=1.4, label='middle of the flat top'); axs[1].loglog(ns, [mid[0] * (n / ns[0]) ** 0.5 for n in ns], color=GREY, lw=0.9, ls='--', label='$\\propto\\sqrt{n}$')
axs[1].set_xlabel('training size n', fontsize=9); axs[1].set_ylabel('k at the flat top', fontsize=8.5); axs[1].legend(fontsize=8, loc='upper left')
axs[1].set_xticks([50, 200, 800, 1600]); axs[1].set_xticklabels(['50', '200', '800', '1600']); axs[1].set_yticks([5, 10, 20, 50, 100]); axs[1].set_yticklabels(['5', '10', '20', '50', '100']); axs[1].minorticks_off()
for ax in axs: ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=8)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s20c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s20c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.7 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.7 / w))

# Problem 56C.1: the two gaps. Writes fig/s56c1.png, verify/c1.out
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch56/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from common import svc_pipe
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
grids = {4: (2.0 ** np.array([3, 6]), 2.0 ** np.array([-10, -7])),
         25: (2.0 ** np.arange(-3, 10, 3), 2.0 ** np.arange(-13, 0, 3)),
         100: (2.0 ** np.linspace(-3, 9, 10), 2.0 ** np.linspace(-13, -1, 10))}
gap = {}
for M, (Cs, Gs) in grids.items():                 # inner score minus fresh accuracy on twelve re-splits
    d = []
    for seed in range(12):
        Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
        gs = GridSearchCV(svc_pipe(), {"svc__C": Cs, "svc__gamma": Gs}, cv=5).fit(Xa, ya)
        d.append(gs.best_score_ - gs.score(Xt, yt))
    gap[M] = np.array(d)
    print("M = %3d cells: best_score_ minus fresh accuracy: mean %+.4f  se %.4f  (positive in %d of 12 splits)" % (M, gap[M].mean(), gap[M].std(ddof=1) / np.sqrt(12), (gap[M] > 0).sum()))

R = np.array(json.load(open('../../ch56/verify/winner.json')))      # the book's twelve re-splits, 25 cells: inner, nested, fresh
dn = R[:, 1] - R[:, 2]; di = R[:, 0] - R[:, 2]
print("\nbook's 25-cell run, twelve re-splits (ch56/verify/winner.json):")
print("  inner - fresh : mean %+.4f  sd %.4f  se %.4f  positive in %d" % (di.mean(), di.std(ddof=1), di.std(ddof=1) / np.sqrt(12), (di > 0).sum()))
print("  nested - fresh: mean %+.4f  sd %.4f  se %.4f  positive in %d   t = %.2f" % (dn.mean(), dn.std(ddof=1), dn.std(ddof=1) / np.sqrt(12), (dn > 0).sum(), dn.mean() / (dn.std(ddof=1) / np.sqrt(12))))
print("  inner - fresh t = %.2f" % (di.mean() / (di.std(ddof=1) / np.sqrt(12))))
print("  one test patient of 143 = %.4f ;  sd of the fresh accuracy over splits %.4f" % (1 / 143, R[:, 2].std(ddof=1)))
print("  nested folds train on 4/5 of 426 = %d rows, the final model on 426" % round(426 * 0.8))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a = axs[0]; Ms = list(grids); m = [gap[k].mean() * 100 for k in Ms]; e = [gap[k].std(ddof=1) / np.sqrt(12) * 100 for k in Ms]
a.bar(range(3), m, yerr=e, color=OR, width=0.6, edgecolor=NAVY, lw=0.5, capsize=2, error_kw=dict(lw=0.8)); a.set_xticks(range(3)); a.set_xticklabels([str(k) for k in Ms])
a.set_xlabel('cells M in the grid'); a.set_ylabel('inner minus fresh (points)'); a.axhline(0, color=GREY, lw=0.6)
a = axs[1]; rng = np.random.RandomState(1)
for j, (v, col, nm) in enumerate(((di, OR, 'inner'), (dn, GREEN, 'nested'))):
    a.plot(j + rng.uniform(-0.12, 0.12, 12), v * 100, 'o', color=col, ms=3, alpha=0.8); a.plot([j - 0.3, j + 0.3], [v.mean() * 100] * 2, color=NAVY, lw=1.4)
a.axhline(0, color=GREY, lw=0.6); a.set_xticks([0, 1]); a.set_xticklabels(['inner − fresh', 'nested − fresh']); a.set_xlim(-0.6, 1.6); a.set_ylabel('points, 25 cells')
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s56c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s56c1.png') as im: w = im.size[0] / 240
print("\nfigure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

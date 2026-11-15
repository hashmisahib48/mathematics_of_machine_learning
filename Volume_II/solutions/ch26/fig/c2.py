# Problem 26C.2: delete the samples that carry more than 5 times the average weight? Writes fig/s26c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

N, M, LIM = 300, 50, 5.0
def weights(m, X, y):                       # the weights after the last round: replay Eq. 26.3-3
    w = np.full(len(y), 1 / len(y))
    for h, a in zip(m.estimators_, m.estimator_weights_):
        w = w * np.exp(a * (h.predict(X) != y)); w /= w.sum()
    return w
def run(seed, d, data_seed=None):
    X, y = make_moons(N + 1000, noise=0.3, random_state=seed); Xt, yt, Xe, ye = X[:N], y[:N], X[N:], y[N:]
    flip = np.random.default_rng(seed).random(N) < 0.10; yn = np.where(flip, 1 - yt, yt)
    mk = lambda: AdaBoostClassifier(DTC(max_depth=d), n_estimators=M, random_state=0)
    m = mk().fit(Xt, yn); w = weights(m, Xt, yn) * N; drop = w > LIM
    m2 = mk().fit(Xt[~drop], yn[~drop])
    return dict(flip=flip, drop=drop, w=w, e0=1 - m.score(Xe, ye), e1=1 - m2.score(Xe, ye), nflip_drop=int((flip & drop).sum()),
                clean_drop=int((~flip & drop).sum()), X=Xt, yn=yn, m2len=len(m2.estimators_))

# the book's single example (seed 3 as in fig/probs.py: make_moons(310) cut to 300)
X, y = make_moons(310, noise=0.3, random_state=3); X, y = X[:300], y[:300]
flip = np.random.default_rng(3).random(300) < 0.10; yn = np.where(flip, 1 - y, y)
m = AdaBoostClassifier(n_estimators=M, random_state=0).fit(X, yn); w = weights(m, X, yn) * N
print("book example: flipped =", int(flip.sum()), " share of weight =", round(float(w[flip].sum() / N), 4), " heaviest = %.1f times the average" % w.max(),
      " flipped among the 30 heaviest =", int(flip[np.argsort(w)[::-1][:30]].sum()))
drop = w > LIM
print("above %.0f times the average: %d samples, of which %d flipped and %d clean (%d of the %d flipped samples are caught)"
      % (LIM, drop.sum(), (flip & drop).sum(), (~flip & drop).sum(), (flip & drop).sum(), flip.sum()))
for k in (3, 5, 8, 12): print("   threshold %2d: %3d removed, %3d flipped" % (k, (w > k).sum(), (flip & (w > k)).sum()))
ex = dict(X=X, yn=yn, flip=flip, drop=drop)

print("\n20 repetitions, 300 training points, 10 per cent flipped, 50 rounds, test on 1000 clean points")
res = {}
for d in (1, 3):
    R = [run(s, d) for s in range(20)]; res[d] = R
    e0 = np.array([r['e0'] for r in R]); e1 = np.array([r['e1'] for r in R])
    print("depth %d: removed per run %.1f (flipped %.1f, clean %.1f; flipped total %.1f) | test error %.3f -> %.3f | better in %d of 20, worse in %d | mean change %+.3f"
          % (d, np.mean([r['drop'].sum() for r in R]), np.mean([r['nflip_drop'] for r in R]), np.mean([r['clean_drop'] for r in R]),
             np.mean([r['flip'].sum() for r in R]), e0.mean(), e1.mean(), (e1 < e0).sum(), (e1 > e0).sum(), (e1 - e0).mean()))
    print("         precision of the rule (share of removed samples that are flipped): %.2f" % (sum(r['nflip_drop'] for r in R) / sum(r['drop'].sum() for r in R)))
# the clean-labels case: the same rule when nothing is wrong
cl = []
for s in range(20):
    Xc, yc = make_moons(N + 1000, noise=0.3, random_state=s); Xt, yt, Xe, ye = Xc[:N], yc[:N], Xc[N:], yc[N:]
    m0 = AdaBoostClassifier(DTC(max_depth=3), n_estimators=M, random_state=0).fit(Xt, yt); dr = weights(m0, Xt, yt) * N > LIM
    m1 = AdaBoostClassifier(DTC(max_depth=3), n_estimators=M, random_state=0).fit(Xt[~dr], yt[~dr])
    cl.append((dr.sum(), 1 - m0.score(Xe, ye), 1 - m1.score(Xe, ye)))
cl = np.array(cl); print("depth 3 with CLEAN labels: the rule still removes %.1f samples per run; test error %.3f -> %.3f" % (cl[:, 0].mean(), cl[:, 1].mean(), cl[:, 2].mean()))

fig, axs = plt.subplots(1, 2, figsize=(4.55, 1.9), dpi=240, gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]; k = ~ex['drop']
ax.scatter(*ex['X'][k & ~ex['flip']].T, s=5, color=BLUE, lw=0); ax.scatter(*ex['X'][k & ex['flip']].T, s=7, color=RED, lw=0)
ax.scatter(*ex['X'][ex['drop'] & ~ex['flip']].T, s=30, facecolor='none', edgecolor=GREEN, lw=1.1)
ax.scatter(*ex['X'][ex['drop'] & ex['flip']].T, s=30, facecolor='none', edgecolor=RED, lw=1.1)
ax.set_xlabel('$x_1$', fontsize=8); ax.set_ylabel('$x_2$', fontsize=8); ax.tick_params(labelsize=7.5)
ax = axs[1]
e0 = np.array([r['e0'] for r in res[3]]); e1 = np.array([r['e1'] for r in res[3]])
for a, b in zip(e0, e1): ax.plot([0, 1], [a, b], color=GREY, lw=0.6, alpha=0.6)
ax.plot([0, 1], [e0.mean(), e1.mean()], color=RED, lw=2.0, marker='o', ms=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['all samples', 'weight above 5 removed'], fontsize=7.5); ax.set_xlim(-0.25, 1.25); ax.tick_params(labelsize=7.5)
ax.set_ylabel('test error (depth 3)', fontsize=8); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.subplots_adjust(wspace=0.3, left=0.08, right=0.99, top=0.97, bottom=0.17)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s26c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s26c2.png') as im: wi = im.size[0] / 240
print("\nfigure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (wi, fs, fs * 4.9 / wi))

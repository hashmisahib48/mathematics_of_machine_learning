# Problem 49C.2: a tree on x and on ln x. Writes fig/s49c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.tree import DecisionTreeRegressor

rng = np.random.RandomState(4)                      # the data of the book's figure
xx = np.sort(rng.lognormal(0, 0.9, 40)); yy = np.where(xx < 1.2, 0.0, 1.0) + 0.25 * np.sin(3 * np.log(xx)) + rng.normal(0, 0.12, 40)
t1 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(xx.reshape(-1, 1), yy)
t2 = DecisionTreeRegressor(max_depth=2, random_state=0).fit(np.log(xx).reshape(-1, 1), yy)
th1 = np.sort(t1.tree_.threshold[t1.tree_.feature >= 0]); th2 = np.sort(np.exp(t2.tree_.threshold[t2.tree_.feature >= 0]))
print("training predictions identical:", np.array_equal(t1.predict(xx.reshape(-1, 1)), t2.predict(np.log(xx).reshape(-1, 1))))
print("thresholds of the tree on x    :", th1.round(4))
print("thresholds of the tree on ln x, back-transformed:", th2.round(4))
gaps = []
for a, b in zip(th1, th2):
    lo = xx[xx < min(a, b)].max(); hi = xx[xx > max(a, b)].min(); gaps.append((lo, hi))
    print("  neighbours %.4f and %.4f: arithmetic midpoint %.4f, geometric midpoint %.4f" % (lo, hi, (lo + hi) / 2, np.sqrt(lo * hi)))
    assert np.isclose((lo + hi) / 2, a) and np.isclose(np.sqrt(lo * hi), b)
gx = np.linspace(xx.min(), xx.max(), 2000)
dif = t1.predict(gx.reshape(-1, 1)) != t2.predict(np.log(gx).reshape(-1, 1))
print("share of a grid of 2000 points where the predictions differ: %.4f" % dif.mean())
print("total length of the disagreement intervals: %.4f of the range %.3f to %.3f (%.4f)" % (sum(abs(a - b) for a, b in zip(th1, th2)), xx.min(), xx.max(), sum(abs(a - b) for a, b in zip(th1, th2)) / (xx.max() - xx.min())))
# the partition of the training samples
leaf = np.bincount(t1.apply(xx.reshape(-1, 1))); print("leaf sizes (on x):", leaf[leaf > 0].tolist(), " same partition on ln x:",
      np.array_equal(t1.apply(xx.reshape(-1, 1)), t2.apply(np.log(xx).reshape(-1, 1))))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.7), dpi=240)
ax = axs[0]; g = np.linspace(xx.min(), xx.max(), 1500)
ax.plot(xx, yy, 'o', color=BLUE, ms=2.6); ax.plot(g, t1.predict(g.reshape(-1, 1)), color=GREEN, lw=1.3, label='tree on $x$'); ax.plot(g, t2.predict(np.log(g).reshape(-1, 1)), color=OR, lw=1.0, ls='--', label='tree on ln $x$')
ax.set_xlabel('$x$', fontsize=9); ax.set_ylabel('$y$', fontsize=9); ax.legend(loc='lower right', fontsize=8, handlelength=1.4); ax.tick_params(labelsize=8); ax.grid(color=GRID, lw=0.5)
ax = axs[1]; lo, hi = gaps[2]; a, b = th1[2], th2[2]; z = np.linspace(lo - 0.05, hi + 0.05, 600)
ax.plot(z, t1.predict(z.reshape(-1, 1)), color=GREEN, lw=1.3); ax.plot(z, t2.predict(np.log(z).reshape(-1, 1)), color=OR, lw=1.0, ls='--')
ax.axvspan(min(a, b), max(a, b), color=PINK); ax.plot([lo, hi], [yy[xx == lo][0], yy[xx == hi][0]], 'o', color=BLUE, ms=4)
ax.set_ylim(0.6, 1.4); ax.text(b - 0.03, 1.28, 'the trees disagree', fontsize=8, ha='right', color=RED)
ax.set_xlabel('$x$ between two neighbouring training values', fontsize=8.5); ax.tick_params(labelsize=8); ax.set_yticks([])
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s49c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s49c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

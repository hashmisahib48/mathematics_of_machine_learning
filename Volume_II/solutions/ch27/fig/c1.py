# Problem 27C.1: stumps cannot represent an exclusive-or. Writes fig/s27c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier as GC

def data(seed, n):
    X = np.random.RandomState(seed).uniform(-1, 1, (n, 2)); return X, ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
X, y = data(0, 240); Xt, yt, Xe, ye = X[:160], y[:160], X[160:], y[160:]
M = {d: GC(max_depth=d, n_estimators=100, random_state=0).fit(Xt, yt) for d in (1, 2)}
for d, m in M.items(): print(f"depth {d}: train {m.score(Xt, yt):.4f}  test {m.score(Xe, ye):.4f}")
print("share of class 1 for x1 > 0: %.3f, for x1 <= 0: %.3f; for x2 > 0: %.3f, x2 <= 0: %.3f" %
      (y[X[:, 0] > 0].mean(), y[X[:, 0] <= 0].mean(), y[X[:, 1] > 0].mean(), y[X[:, 1] <= 0].mean()))
# additivity: F(a,b) - F(a,0) - F(0,b) + F(0,0) for the stump model on a grid
g = np.linspace(-.9, .9, 19); G = np.array([[a, b] for a in g for b in g]); F = M[1].decision_function(G).reshape(19, 19)
inter = F - F[:, [9]] - F[[9], :] + F[9, 9]; print("stumps: largest |F(a,b) - F(a,c) - F(c,b) + F(c,c)| =", np.abs(inter).max().round(10))
F2 = M[2].decision_function(G).reshape(19, 19); print("depth 2: same quantity =", np.abs(F2 - F2[:, [9]] - F2[[9], :] + F2[9, 9]).max().round(3))
# 20 repetitions with fresh test points
a1, a2 = [], []
for s in range(20):
    Xa, ya = data(100 + s, 160); Xb, yb = data(500 + s, 2000)
    a1.append(GC(max_depth=1, n_estimators=100, random_state=0).fit(Xa, ya).score(Xb, yb)); a2.append(GC(max_depth=2, n_estimators=100, random_state=0).fit(Xa, ya).score(Xb, yb))
print("20 fresh training sets, 2000 fresh test points: stumps %.3f (range %.3f - %.3f), depth 2 %.3f" % (np.mean(a1), min(a1), max(a1), np.mean(a2)))
# a remedy: give the stumps the product feature
Xp = np.c_[X, X[:, 0] * X[:, 1]]; mp = GC(max_depth=1, n_estimators=100, random_state=0).fit(Xp[:160], yt)
print("stumps with the extra feature x1*x2: test accuracy %.4f" % mp.score(Xp[160:], ye))
fig, axs = plt.subplots(1, 2, figsize=(4.3, 2.0), dpi=240); gg = np.linspace(-1, 1, 120); XX = np.array([[a, b] for b in gg for a in gg])
for ax, d in zip(axs, (1, 2)):
    ax.contourf(gg, gg, M[d].predict_proba(XX)[:, 1].reshape(120, 120), levels=np.linspace(0, 1, 11), cmap='RdBu_r', alpha=0.8)
    ax.scatter(Xt[:, 0], Xt[:, 1], c=np.where(yt == 1, OR, BLUE), s=5, lw=0, zorder=5); ax.set_aspect('equal'); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=7.5); ax.set_xlabel('$x_1$', fontsize=8); ax.set_title('max_depth = %d, test %.3f' % (d, M[d].score(Xe, ye)), fontsize=8, pad=3)
axs[0].set_ylabel('$x_2$', fontsize=8); fig.subplots_adjust(wspace=0.35, left=0.08, right=0.99, top=0.88, bottom=0.17)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s27c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s27c1.png') as im: wi = im.size[0] / 240
print("figure width %.2f in, smallest font %.1f pt -> printed at 4.5 in: %.1f pt" % (wi, fs, fs * 4.5 / wi))

# Problem 24C.1: a tree cannot extrapolate. Writes fig/s24c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR

# the book's problem: 40 points of the exact line y = 2x + 1 on [0, 5]
x = np.linspace(0, 5, 40); y = 2 * x + 1; X = x[:, None]
tree = DTR(random_state=0).fit(X, y)
print("leaves %d, depth %d; spacing of the points %.4f" % (tree.get_n_leaves(), tree.get_depth(), x[1] - x[0]))
for v in (5, 6, 8, 100): print("x = %5.1f  tree %.3f  line %.3f" % (v, tree.predict([[v]])[0], 2 * v + 1))
thr = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0]); print("last two thresholds %.4f %.4f; last leaf value %.3f" % (thr[-2], thr[-1], tree.predict([[5.0]])[0]))
print("largest value the tree can ever return: %.3f = max of the training targets %.3f" % (tree.tree_.value[tree.tree_.feature < 0].max(), y.max()))
xs = np.linspace(0, 8, 2001); pr = tree.predict(xs[:, None])
print("error inside [0, 5]: max %.4f (half a step of the data is %.4f);  error at x = 8: %.3f" % (np.abs(pr[xs <= 5] - (2 * xs[xs <= 5] + 1)).max(), 2 * (x[1] - x[0]) / 2, 17 - pr[-1]))
rf = RFR(100, random_state=0).fit(X, y); print("forest of 100 trees at x = 8: %.3f" % rf.predict([[8.0]])[0])
lr = LR().fit(X, y); print("linear regression at x = 8: %.3f" % lr.predict([[8.0]])[0])

# a curve that is only roughly linear: the tree on the residuals of a line extrapolates
rng = np.random.default_rng(0); x2 = np.sort(rng.uniform(0, 5, 80)); X2 = x2[:, None]
f = lambda t: 2 * t + 1 + 0.6 * np.sin(2 * t); y2 = f(x2) + rng.normal(0, 0.15, 80)
t2 = DTR(min_samples_leaf=3, random_state=0).fit(X2, y2)
l2 = LR().fit(X2, y2); comb = DTR(min_samples_leaf=3, random_state=0).fit(X2, y2 - l2.predict(X2))
pred_c = lambda z: l2.predict(z) + comb.predict(z)
xo = np.linspace(5, 8, 301); Xo = xo[:, None]
rm = lambda p: np.sqrt(np.mean((p - f(xo)) ** 2))
print("noisy curve, RMSE on x in [5, 8]: tree %.3f   line + tree on residuals %.3f   line alone %.3f" % (rm(t2.predict(Xo)), rm(pred_c(Xo)), rm(l2.predict(Xo))))
print("prediction of the tree at x = 8: %.3f; max of its training targets %.3f" % (t2.predict([[8.0]])[0], y2.max()))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.0), dpi=240)
ax = axs[0]; ax.plot(xs, 2 * xs + 1, color=GREY, lw=1.0, ls=':', label='line $2x+1$'); ax.plot(xs, pr, color=GREEN, lw=1.6, label='tree'); ax.scatter(x[::3], y[::3], s=9, color=BLUE, zorder=5)
ax.axvline(5, color=PUR, lw=0.9, ls='--'); ax.annotate('', xy=(8, pr[-1]), xytext=(8, 17), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.0)); ax.text(7.85, 14, 'error 6', color=RED, fontsize=8.6, ha='right', va='center')
ax.set_title('the exact line of the problem', fontsize=8.6, pad=3); ax.set_xlabel('$x$', fontsize=8.6); ax.set_ylabel('$y$', fontsize=8.6); ax.legend(loc='upper left', fontsize=8.6, borderaxespad=0.1)
ax = axs[1]; xx = np.linspace(0, 8, 400)[:, None]; ax.plot(xx, f(xx[:, 0]), color=GREY, lw=1.0, ls=':', label='truth'); ax.scatter(x2, y2, s=5, color=BLUE, zorder=5, lw=0)
ax.plot(xx, t2.predict(xx), color=GREEN, lw=1.5, label='tree'); ax.plot(xx, pred_c(xx), color=OR, lw=1.5, label='line + tree'); ax.axvline(5, color=PUR, lw=0.9, ls='--')
ax.set_title('a noisy curve', fontsize=8.6, pad=3); ax.set_xlabel('$x$', fontsize=8.6); ax.legend(loc='upper left', fontsize=8.6, borderaxespad=0.1)
for a in axs: a.tick_params(labelsize=8.6); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
fig.subplots_adjust(wspace=0.22)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s24c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s24c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

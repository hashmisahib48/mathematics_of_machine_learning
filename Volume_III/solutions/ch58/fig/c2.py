# Problem 58C.2: a flat partial dependence hides an interaction. Writes fig/s58c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.inspection import partial_dependence, permutation_importance

rg = np.random.RandomState(3); X = rg.uniform(-1, 1, (600, 2)); y = ((X[:, 0] * X[:, 1] + 0.1 * rg.standard_normal(600)) > 0).astype(int)
gb = GBC(random_state=0).fit(X, y)
print("class share %.3f, training accuracy %.4f" % (y.mean(), gb.score(X, y)))
pd_ = partial_dependence(gb, X, [0], kind='both', grid_resolution=40, method='brute', response_method='predict_proba')
g, ice, avg = pd_['grid_values'][0], pd_['individual'][0], pd_['average'][0]
pos, neg = X[:, 1] > 0, X[:, 1] < 0
mp, mn = ice[pos].mean(0), ice[neg].mean(0)
print("PD of x1: min %.3f max %.3f (range %.3f)" % (avg.min(), avg.max(), avg.max() - avg.min()))
print("mean ICE curve of the rows with x2 > 0 (%d rows): %.3f at x1 = %.2f to %.3f at x1 = %.2f" % (pos.sum(), mp[0], g[0], mp[-1], g[-1]))
print("mean ICE curve of the rows with x2 < 0 (%d rows): %.3f to %.3f" % (neg.sum(), mn[0], mn[-1]))
print("PD = share-weighted mean of the two: max difference %.1e" % np.abs(avg - (pos.mean() * mp + neg.mean() * mn)).max())
rngc = ice.max(1) - ice.min(1); print("range of an ICE curve: mean %.3f, smallest %.3f; share of curves with range > 0.5: %.3f" % (rngc.mean(), rngc.min(), (rngc > 0.5).mean()))
xt = np.random.RandomState(4).uniform(-1, 1, (1000, 2)); yt = ((xt[:, 0] * xt[:, 1] + 0.1 * np.random.RandomState(5).standard_normal(1000)) > 0).astype(int)
pi_ = permutation_importance(gb, xt, yt, n_repeats=20, random_state=0)
print("held-out accuracy %.3f; permutation importance (accuracy): x1 %.3f  x2 %.3f" % (gb.score(xt, yt), pi_.importances_mean[0], pi_.importances_mean[1]))
# two-way view: the model on a grid
gx = np.linspace(-1, 1, 41); G1, G2 = np.meshgrid(gx, gx); Z = gb.predict_proba(np.c_[G1.ravel(), G2.ravel()])[:, 1].reshape(G1.shape)
q = lambda a, b: Z[(np.sign(G1) == a) & (np.sign(G2) == b)].mean()
print("mean P(class 1) on the grid in the quadrants (x1+,x2+) %.3f (x1-,x2+) %.3f (x1+,x2-) %.3f (x1-,x2-) %.3f" % (q(1, 1), q(-1, 1), q(1, -1), q(-1, -1)))
# a model that ignores x1 for comparison
g2 = GBC(random_state=0).fit(X[:, [1]], y); pd2 = partial_dependence(g2, X[:, [1]], [0], grid_resolution=40, method='brute', response_method='predict_proba')['average'][0]
print("model fitted with x2 only: accuracy %.3f (chance 0.5); its PD of x2 ranges from %.3f to %.3f" % (g2.score(X[:, [1]], y), pd2.min(), pd2.max()))

OR_, BL_ = '#E8871E', '#1F5FBF'
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
a = axs[0]; idx = np.random.RandomState(0).choice(600, 40, replace=False)
for i in idx: a.plot(g, ice[i], color=OR if X[i, 1] > 0 else BLUE, lw=0.4, alpha=0.45)
a.plot(g, mp, color=OR, lw=1.6, ls='--'); a.plot(g, mn, color=BLUE, lw=1.6, ls='-.'); a.plot(g, avg, color=GREEN, lw=2.4)
a.text(0.38, 0.6, 'PD', color=GREEN, fontsize=8.4, weight='bold')
a.set_xlabel('$x_1$'); a.set_ylabel('$P$(class 1)'); a.set_ylim(-0.03, 1.03); a.set_xticks([-1, 0, 1]); a.set_yticks([0, 0.5, 1])
a = axs[1]; cm = LinearSegmentedColormap.from_list('bo', [BLUE, '#F4F6FA', OR])
im = a.imshow(Z, origin='lower', extent=[-1, 1, -1, 1], cmap=cm, vmin=0, vmax=1, aspect='auto'); a.set_xlabel('$x_1$'); a.set_ylabel('$x_2$'); a.set_xticks([-1, 0, 1]); a.set_yticks([-1, 0, 1])
cb = fig.colorbar(im, ax=a, pad=0.03, fraction=0.06); cb.set_ticks([0, 0.5, 1]); cb.ax.tick_params(labelsize=8.2); cb.set_label('$P$(class 1)', fontsize=8.6)
for a in axs: a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
axs[0].grid(color=GRID, lw=0.5); axs[0].set_axisbelow(True)
fig.subplots_adjust(wspace=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s58c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s58c2.png') as im_: w = im_.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

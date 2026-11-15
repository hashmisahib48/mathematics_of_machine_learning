# Problem 27C.2: the out-of-bag loss is too optimistic. Writes fig/s27c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor as GR
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=600, noise=1.0, random_state=0); Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.5, random_state=0)
kw = dict(learning_rate=0.1, max_depth=3, subsample=0.5, random_state=0)
g = GR(n_estimators=300, **kw).fit(Xt, yt); te = np.array([mse(ye, p) for p in g.staged_predict(Xe)]); tr_all = np.array([mse(yt, p) for p in g.staged_predict(Xt)])
oob = g.oob_scores_; print("at 300 trees: out-of-bag %.4f, in-bag (train_score_) %.4f, all training samples %.4f, test %.3f; noise variance of the targets 1.0" % (oob[-1], g.train_score_[-1], tr_all[-1], te[-1]))
print("at  50 trees: out-of-bag %.3f, in-bag %.3f, test %.3f" % (oob[49], g.train_score_[49], te[49]))
print("out-of-bag loss still falling at 300: smallest at tree %d; test error smallest at tree %d (%.3f), test at 100: %.3f, at 300: %.3f" % (oob.argmin() + 1, te.argmin() + 1, te.min(), te[99], te[-1]))
print("expected number of the 299 earlier trees that used a given out-of-bag sample: %.1f" % (299 * 0.5))
# an honest rule: hold back 25 per cent of the training part
Xa, Xv, ya, yv = train_test_split(Xt, yt, test_size=0.25, random_state=1)
h = GR(n_estimators=300, **kw).fit(Xa, ya); va = np.array([mse(yv, p) for p in h.staged_predict(Xv)]); teh = np.array([mse(ye, p) for p in h.staged_predict(Xe)])
print("validation rule (225 / 75 split): best validation stage %d (validation MSE %.3f); test MSE there %.3f; best test MSE of this model %.3f at %d" % (va.argmin() + 1, va.min(), teh[va.argmin()], teh.min(), teh.argmin() + 1))
es = GR(n_estimators=1000, learning_rate=0.1, max_depth=3, subsample=0.5, n_iter_no_change=10, random_state=0).fit(Xt, yt)
print("n_iter_no_change=10 (validation_fraction 0.1): keeps %d trees, test MSE %.3f" % (es.n_estimators_, mse(ye, es.predict(Xe))))
fig, ax = plt.subplots(figsize=(3.6, 1.9), dpi=240); k = np.arange(1, 301)
ax.plot(k, te, color=GREY, lw=1.4, label='test'); ax.plot(k, oob, color=RED, lw=1.4, ls=':', label='out-of-bag'); ax.plot(k, g.train_score_, color=BLUE, lw=1.0, label='in-bag')
ax.plot(k, va, color=GREEN, lw=1.2, ls='--', label='validation'); ax.axhline(1.0, color=PUR, lw=0.8, ls='-.'); ax.text(205, 1.12, 'noise level 1', fontsize=7.5, color=PUR)
ax.set_yscale('log'); ax.set_xlabel('trees', fontsize=8); ax.set_ylabel('mean squared error', fontsize=8); ax.tick_params(labelsize=7.5); ax.legend(fontsize=7.5, ncol=2, loc='lower left', handlelength=1.6, columnspacing=0.8, borderaxespad=0.1)
ax.set_ylim(0.01, 12); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s27c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s27c2.png') as im: wi = im.size[0] / 240
print("figure width %.2f in, smallest font %.1f pt -> printed at 3.7 in: %.1f pt" % (wi, fs, fs * 3.7 / wi))

# Problem 25C.1: two copies of a feature. Writes fig/s25c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import RandomForestClassifier as RFC

rng = np.random.default_rng(0); n = 1000                                     # the book's data
s_ = rng.integers(0, 3, n); y_ = rng.random(n) < 0.2 + 0.3 * s_; u_ = rng.uniform(size=n)
X1 = np.c_[s_, u_]; X2 = np.c_[s_, s_, u_]
f1 = RFC(300, random_state=0).fit(X1[:500], y_[:500]); f2 = RFC(300, random_state=0).fit(X2[:500], y_[:500])
print("importances, one copy :", f1.feature_importances_.round(3).tolist(), " test accuracy %.3f" % f1.score(X1[500:], y_[500:]))
print("importances, two copies:", f2.feature_importances_.round(3).tolist(), " test accuracy %.3f" % f2.score(X2[500:], y_[500:]))
print("signal total with two copies: %.3f (one copy: %.3f)" % (f2.feature_importances_[:2].sum(), f1.feature_importances_[0]))
print("P(class 1) is 0.2, 0.5, 0.8 for s = 0, 1, 2; the best possible accuracy is %.3f; share of class 1: %.3f" % ((0.8 + 0.5 + 0.8) / 3, y_.mean()))
print("features searched per split: %d (of 2 columns) and %d (of 3 columns)" % (f1.estimators_[0].max_features_, f2.estimators_[0].max_features_))
cnt = np.zeros(3)
for t in f2.estimators_:
    for fe in t.tree_.feature[t.tree_.feature >= 0]: cnt[fe] += 1
print("questions asked about s, its copy, u (300 trees): %s  -> shares %s" % (cnt.astype(int).tolist(), (cnt / cnt.sum()).round(3).tolist()))
cnt1 = np.zeros(2)
for t in f1.estimators_:
    for fe in t.tree_.feature[t.tree_.feature >= 0]: cnt1[fe] += 1
print("questions asked about s, u with one copy: %s" % cnt1.astype(int).tolist())
# permutation importance on the other half, 40 shufflings
Xt, yt = X2[500:], y_[500:]; base = f2.score(Xt, yt); r = np.random.default_rng(1); res = {}
for name, cols in (("s", [0]), ("copy", [1]), ("s and copy together", [0, 1]), ("u", [2])):
    d = []
    for _ in range(40):
        Z = Xt.copy(); Z[:, cols] = Z[r.permutation(len(Z))][:, cols]; d.append(base - (f2.predict(Z) == yt).mean())
    res[name] = (np.mean(d), np.std(d)); print("permutation importance of %-20s %.4f +- %.4f" % (name, *res[name]))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.95), dpi=240)
ax = axs[0]; xx = np.arange(3); ax.bar(xx[[0, 2]] - 0.2, f1.feature_importances_, 0.38, color=GREY, label='one copy'); ax.bar(xx + 0.2, f2.feature_importances_, 0.38, color=GREEN, label='two copies')
ax.set_xticks(xx); ax.set_xticklabels(['$s$', 'copy', '$u$'], fontsize=8.6); ax.set_ylabel('impurity importance', fontsize=8.6); ax.set_title('impurity importance', fontsize=8.6, pad=3); ax.legend(loc='upper left', fontsize=8.6, borderaxespad=0.1)
ax.set_ylim(0, 0.95)
ax = axs[1]; ks = list(res); ax.bar(range(4), [res[k][0] for k in ks], 0.55, yerr=[res[k][1] for k in ks], color=[BLUE, BLUE, OR, GREY], capsize=2, error_kw=dict(lw=0.8))
ax.set_xticks(range(4)); ax.set_xticklabels(['$s$', 'copy', 'both', '$u$'], fontsize=8.6); ax.axhline(0, color=GREY, lw=0.7); ax.set_title('permutation importance', fontsize=8.6, pad=3); ax.set_ylabel('loss of accuracy', fontsize=8.6)
for a in axs: a.tick_params(labelsize=8.6); a.grid(color=GRID, lw=0.5, axis='y'); a.set_axisbelow(True)
fig.subplots_adjust(wspace=0.35)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s25c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s25c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

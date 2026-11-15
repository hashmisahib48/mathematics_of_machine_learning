# Problem 53C.1: an exclusive-or pair among noise columns. Writes fig/s53c1.png, prints success shares.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, SequentialFeatureSelector, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def data(seed, n=200, noise=8):
    rng = np.random.RandomState(seed); u = rng.uniform(-1, 1, (n, 2)); y = ((u[:, 0] > 0) ^ (u[:, 1] > 0)).astype(int)
    return np.c_[u, rng.uniform(-1, 1, (n, noise))], y
# the book's data (seed 2, two columns): F of each column alone
X2, y2 = data(2, noise=0); print("book data: F =", f_classif(X2, y2)[0].round(4).tolist())
# the pair itself, with a linear and with a nonlinear model
cvl = cross_val_score(LogisticRegression(), X2, y2, cv=5).mean(); cvt = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), X2, y2, cv=5).mean()
print("CV accuracy on the pair: logistic regression %.3f, tree of depth 4 %.3f" % (cvl, cvt))
methods = {
 'F filter, k = 2':           lambda: SelectKBest(f_classif, k=2),
 'RFE, logistic':             lambda: RFE(LogisticRegression(), n_features_to_select=2),
 'forward, logistic':         lambda: SequentialFeatureSelector(LogisticRegression(), n_features_to_select=2, cv=5),
 'forward, tree':             lambda: SequentialFeatureSelector(DecisionTreeClassifier(max_depth=4, random_state=0), n_features_to_select=2, cv=5),
 'RFE, random forest':        lambda: RFE(RandomForestClassifier(40, random_state=0), n_features_to_select=2),
}
reps = 20; hit = {k: 0 for k in methods}; first = 0
for seed in range(reps):
    X, y = data(100 + seed)
    for k, mk in methods.items():
        s = mk().fit(X, y).get_support(); hit[k] += int(s[0] and s[1])
share = {k: v / reps for k, v in hit.items()}
print("10 columns (2 useful, 8 noise), 200 rows, %d repetitions: share of runs that keep columns 0 and 1" % reps)
for k, v in share.items(): print("  %-20s %2d of %d  = %.2f" % (k, hit[k], reps, v))
chance = 1 / 45; print("a random pair of the 10 columns is right with probability 1/45 = %.3f" % chance)
Xn, yn = data(100); Fn, _ = f_classif(Xn, yn); print("F of the ten columns in repetition 0:", Fn.round(2).tolist()); print("rank by F of the two useful columns:", [int((Fn > Fn[j]).sum()) + 1 for j in (0, 1)])

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.75), dpi=240, gridspec_kw={'width_ratios': [1, 1.25]})
a = axs[0]; a.bar(range(10), Fn, color=[OR, OR] + [BLUE] * 8, width=0.75); a.set_xlabel('column (orange: the pair)'); a.set_ylabel('F score'); a.set_xticks(range(0, 10, 3))
a = axs[1]; names = list(methods)[::-1]; a.barh(range(5), [share[k] for k in names], color=[GREEN if share[k] > 0.5 else RED for k in names], height=0.6)
a.set_yticks(range(5)); a.set_yticklabels(names); a.set_xlim(0, 1.05); a.set_xlabel('share of runs that find the pair')
for j, k in enumerate(names): a.text(share[k] + 0.02, j, '%.2f' % share[k], va='center', fontsize=8.2)
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.95)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s53c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s53c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

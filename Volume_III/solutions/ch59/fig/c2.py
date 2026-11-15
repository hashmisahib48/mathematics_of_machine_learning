# Problem 59C.2: four causes of a cross-validated AUC of 0.998, each tested. Writes fig/s59c2.png, prints verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# the chapter's customers (800, 20 per cent cancel); four columns, one of them a leak
r_ = np.random.RandomState(0); N = 800; y = (r_.rand(N) < 0.2).astype(int)
cols = {'visits': r_.normal(0, 1, N) + 0.5 * y, 'tenure': r_.normal(0, 1, N) - 0.3 * y,
        'spend': r_.normal(0, 1, N) + 0.2 * y, 'status': y * 1.0 + r_.normal(0, 0.05, N)}
X = np.c_[[cols[k] for k in cols]].T
rf = lambda: RF(100, random_state=0)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
auc = lambda m, X_, y_, cv_=cv, g=None: cross_val_score(m, X_, y_, cv=cv_, groups=g, scoring='roc_auc').mean()
R = {}
print("== Test 1: one column at a time (AUC of the column itself)")
for j, k in enumerate(cols):
    a = roc_auc_score(y, X[:, j]); print("  %-7s %.3f" % (k, max(a, 1 - a)))
R['all four columns'] = auc(rf(), X, y); R['without status'] = auc(rf(), X[:, :3], y)
print("== Test 2: drop the suspect column and refit: all columns %.3f, without 'status' %.3f" % (R['all four columns'], R['without status']))

print("== Test 3: duplicated rows (every customer entered twice), random folds against folds by customer")
X2 = np.r_[X[:, :3], X[:, :3]]; y2 = np.r_[y, y]; g2 = np.r_[np.arange(N), np.arange(N)]
R['rows twice, random folds'] = auc(rf(), X2, y2, StratifiedKFold(5, shuffle=True, random_state=0))
R['rows twice, by customer'] = auc(rf(), X2, y2, GroupKFold(5), g2)
print("  random folds %.3f   by customer %.3f   (single copy: %.3f)" % (R['rows twice, random folds'], R['rows twice, by customer'], R['without status']))

print("== Test 4: a column filter fitted on all rows before cross-validation (200 customers, 500 noise columns, labels unrelated)")
out, ins = [], []
for seed in range(10):                      # ten invented tables, to average out the luck of one
    rn = np.random.RandomState(seed); Xn = rn.normal(0, 1, (200, 500)); yn = (rn.rand(200) < 0.2).astype(int)
    sel = SelectKBest(f_classif, k=10).fit(Xn, yn).transform(Xn)
    out.append(auc(RF(100, random_state=0), sel, yn))
    ins.append(auc(make_pipeline(SelectKBest(f_classif, k=10), RF(100, random_state=0)), Xn, yn))
R['filter outside CV'] = np.mean(out); R['filter inside CV'] = np.mean(ins)
print("  filter outside the folds %.3f (sd %.3f)   inside the pipeline %.3f (sd %.3f)   (true value 0.5), mean of 10 tables"
      % (np.mean(out), np.std(out), np.mean(ins), np.std(ins)))

order = ['all four columns', 'without status', 'rows twice, random folds', 'rows twice, by customer', 'filter outside CV', 'filter inside CV']
lab = ['all four columns', 'without "status"', 'rows twice, random folds', 'rows twice, by customer', 'filter outside folds', 'filter inside pipeline']
cc = [RED, GREEN, RED, GREEN, RED, GREEN]
fig, ax = plt.subplots(figsize=(4.9, 1.9), dpi=240)
ys = np.arange(6)[::-1]
ax.barh(ys, [R[k] for k in order], color=cc, edgecolor=NAVY, lw=0.5, height=0.62)
for yy, k in zip(ys, order): ax.text(R[k] + 0.012, yy, '%.3f' % R[k], va='center', fontsize=8, color=NAVY)
ax.set_yticks(ys); ax.set_yticklabels(lab, fontsize=8); ax.set_xlim(0.4, 1.1); ax.set_xticks([0.5, 0.75, 1.0]); ax.tick_params(labelsize=8)
ax.set_xlabel('cross-validated AUC', fontsize=8); ax.axvline(0.5, color=GREY, lw=0.6); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s59c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s59c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

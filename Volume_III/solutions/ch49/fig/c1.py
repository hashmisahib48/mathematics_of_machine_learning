# Problem 49C.1: why scaling before cross-validation leaks little, and what leaks a lot. Writes fig/s49c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
# 1. how far can a training fold's mean move when the test fold is added? (in units of the standard deviation)
for n in (569, 50):
    idx = np.random.RandomState(0).permutation(len(y))[:n]; Xs = X[idx]; shifts = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(Xs):
        shifts.append((np.abs(Xs.mean(0) - Xs[tr].mean(0)) / Xs.std(0)).mean())
    print("n = %3d: mean shift of the fold mean, in standard deviations: %.4f   (theory sqrt(1/n_tr - 1/n) = %.4f)"
          % (n, np.mean(shifts), np.sqrt(1 / (0.8 * n) - 1 / n)))
# 2. leaky minus honest accuracy: a scaler against a selection by the target (pure noise: 60 rows, 500 columns)
reps = 30; d_scale = []; d_sel = []; acc_leak = []; acc_hon = []
for sd in range(reps):
    rng = np.random.RandomState(sd); Xn = rng.normal(size=(60, 500)); yn = rng.randint(0, 2, 60)
    lr = lambda: LogisticRegression(max_iter=2000)
    cv = KFold(5, shuffle=True, random_state=0)
    hon = cross_val_score(make_pipeline(SelectKBest(f_classif, k=10), lr()), Xn, yn, cv=cv).mean()
    leak = cross_val_score(lr(), SelectKBest(f_classif, k=10).fit_transform(Xn, yn), yn, cv=cv).mean()
    hs = cross_val_score(make_pipeline(StandardScaler(), lr()), Xn, yn, cv=cv).mean()
    ls = cross_val_score(lr(), StandardScaler().fit_transform(Xn), yn, cv=cv).mean()
    d_sel.append(leak - hon); d_scale.append(ls - hs); acc_leak.append(leak); acc_hon.append(hon)
print("noise data, 60 rows x 500 columns, labels random (true accuracy 0.5), %d repetitions:" % reps)
print("  selection of 10 columns by F-score on all rows: leaky accuracy %.3f, honest %.3f (difference %.3f, smallest %.3f)" % (np.mean(acc_leak), np.mean(acc_hon), np.mean(d_sel), np.min(d_sel)))
print("  StandardScaler on all rows:                     difference leaky - honest: mean %.4f, sd %.4f, largest |difference| %.3f" % (np.mean(d_scale), np.std(d_scale), np.max(np.abs(d_scale))))
# 3. the same for the scaler on real data, 20 subsets of 50 rows (as in the book's figure)
dr = []
for sd in range(20):
    idx = np.random.RandomState(sd).permutation(len(y))[:50]
    dr.append(cross_val_score(SVC(), StandardScaler().fit_transform(X[idx]), y[idx], cv=5).mean() - cross_val_score(make_pipeline(StandardScaler(), SVC()), X[idx], y[idx], cv=5).mean())
dr = np.array(dr); print("breast cancer, 20 subsets of 50 rows, StandardScaler + SVC: mean %.4f, %d of 20 differ from zero, range %.2f to %.2f" % (dr.mean(), (np.abs(dr) > 1e-9).sum(), dr.min(), dr.max()))

fig, ax = plt.subplots(figsize=(5.0, 1.6), dpi=240)
rng = np.random.RandomState(0)
for k, (v, col, lab) in enumerate(((d_scale, BLUE, 'StandardScaler on all rows'), (dr, GREEN, 'same, breast cancer (50 rows)'), (d_sel, RED, 'selection by the target'))):
    ax.plot(np.array(v), k + rng.uniform(-0.18, 0.18, len(v)), 'o', color=col, ms=3, alpha=0.8); ax.plot([np.mean(v)] * 2, [k - 0.3, k + 0.3], color=NAVY, lw=1.6)
ax.axvline(0, color=GREY, lw=0.8); ax.set_yticks(range(3)); ax.set_yticklabels(['StandardScaler\n(noise data)', 'StandardScaler\n(breast cancer)', 'selection of 10\ncolumns by F-score'], fontsize=8)
ax.set_xlabel('leaky score minus honest score (cross-validated accuracy)', fontsize=8.5); ax.tick_params(labelsize=8); ax.grid(color=GRID, lw=0.5, axis='x'); ax.set_axisbelow(True); ax.set_ylim(-0.6, 2.6)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s49c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s49c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

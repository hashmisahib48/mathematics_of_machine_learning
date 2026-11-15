# Problem 24C.2: ten seeds and one test set. Writes fig/s24c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.stats import binomtest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC

d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0, stratify=d.target)
n = len(yte); preds = {s: DTC(random_state=s).fit(Xtr, ytr).predict(Xte) for s in range(10)}
acc = np.array([(preds[s] == yte).mean() for s in range(10)]); wrong = np.array([(preds[s] != yte).sum() for s in range(10)])
print("test accuracy per seed:", acc.round(4).tolist()); print("test errors per seed  :", wrong.tolist(), " (of %d)" % n)
print("range %.4f - %.4f = %d patients = %.4f; mean %.4f, sd %.4f" % (acc.min(), acc.max(), wrong.max() - wrong.min(), acc.max() - acc.min(), acc.mean(), acc.std(ddof=1)))
se = np.sqrt(acc.mean() * (1 - acc.mean()) / n); print("binomial standard error of one accuracy on %d patients: %.4f; of a difference of two independent ones: %.4f" % (n, se, np.sqrt(2) * se))
a, b = int(np.argmax(acc)), int(np.argmin(acc)); print("colleague's tree: seed %d (%.4f); competitor's: seed %d (%.4f)" % (a, acc[a], b, acc[b]))
ca, cb = preds[a] == yte, preds[b] == yte; n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
print("patients right only for hers: %d, only for his: %d, disagreement between the two trees %d patients" % (n10, n01, int((preds[a] != preds[b]).sum())))
print("exact McNemar (binomial) p-value: %.3f" % binomtest(n10, n10 + n01, 0.5).pvalue)
# the same two seeds on 200 other train/test splits
diff = []; both = []
for r in range(200):
    A_, B_, ya, yb = train_test_split(d.data, d.target, test_size=0.25, random_state=100 + r, stratify=d.target)
    s1 = DTC(random_state=a).fit(A_, ya).score(B_, yb); s2 = DTC(random_state=b).fit(A_, ya).score(B_, yb); diff.append(s1 - s2); both.append((s1, s2))
diff = np.array(diff); both = np.array(both)
print("200 splits: mean accuracy seed %d %.4f, seed %d %.4f" % (a, both[:, 0].mean(), b, both[:, 1].mean()))
print("200 splits: difference mean %.4f, sd %.4f, standard error %.4f; seed %d better in %d splits, worse in %d, equal in %d" % (diff.mean(), diff.std(ddof=1), diff.std(ddof=1) / np.sqrt(200), a, (diff > 0).sum(), (diff < 0).sum(), (diff == 0).sum()))
print("difference observed on the book's split: %.4f = %.2f sd of the 200 differences" % (acc[a] - acc[b], (acc[a] - acc[b]) / diff.std(ddof=1)))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.95), dpi=240)
ax = axs[0]; ax.axhspan(acc.mean() - se, acc.mean() + se, color=LB, zorder=0); ax.axhline(acc.mean(), color=GREEN, lw=1.0, ls='--')
ax.plot(range(10), acc, 'o', color=BLUE, ms=4.5); ax.plot([a, b], [acc[a], acc[b]], 'o', color=OR, ms=6, mfc='none', mew=1.3)
ax.set_xlabel('random_state of the tree', fontsize=8.6); ax.set_ylabel('test accuracy', fontsize=8.6); ax.set_title('one split, ten seeds', fontsize=8.6, pad=3); ax.set_xticks([0, 3, 6, 9]); ax.set_ylim(0.87, 0.95)
ax.text(9.3, acc.mean() + se + 0.002, '±1 s.e.', fontsize=8.6, color=NAVY, ha='right', va='bottom')
ax = axs[1]; ax.hist(diff, bins=np.arange(-0.0805, 0.0806, 0.007), color=BLUE, edgecolor='white'); ax.axvline(0, color=GREY, lw=0.8); ax.axvline(acc[a] - acc[b], color=OR, lw=1.4)
ax.text(acc[a] - acc[b] + 0.004, ax.get_ylim()[1] * 0.92, 'her result', color=OR, fontsize=8.6, va='top'); ax.set_xlabel('accuracy of seed %d minus seed %d' % (a, b), fontsize=8.6); ax.set_ylabel('splits', fontsize=8.6); ax.set_title('200 other splits', fontsize=8.6, pad=3)
for q in axs: q.tick_params(labelsize=8.6); q.grid(color=GRID, lw=0.5); q.set_axisbelow(True)
fig.subplots_adjust(wspace=0.3)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s24c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s24c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

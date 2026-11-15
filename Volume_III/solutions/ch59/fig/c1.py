# Problem 59C.1: readmissions, invented hospital table. Why the protocol decides the number.
# Writes fig/s59c1.png and prints verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

rng = np.random.RandomState(0)
P = 6000                                          # patients
n_st = 1 + rng.poisson(3.0, P)                    # stays per patient
pid = np.repeat(np.arange(P), n_st); N = len(pid)
frail = rng.normal(0, 1, P)                       # hidden frailty of the patient
height = np.round(rng.normal(170, 9, P), 1); birth = rng.randint(1930, 2000, P) + rng.rand(P).round(2)
year = rng.rand(N) * 3                            # date of the stay in years 0..3
age = 2020 - np.floor(birth[pid]) + year
los = rng.gamma(2.0, 2.0, N) + 1.0 * frail[pid]   # length of stay
prior = np.array([np.arange(k) for k in n_st], dtype=object); prior = np.concatenate(prior)   # earlier stays
z = -7.9 + 3.2 * frail[pid] + 0.12 * (los - 4) + 0.15 * prior
y = (rng.rand(N) < 1 / (1 + np.exp(-z))).astype(int)
Xf = np.c_[age, los, prior, height[pid], birth[pid]]
print("stays %d, patients %d, return rate %.4f (%d returns)" % (N, P, y.mean(), y.sum()))

def score(tr, te, model):
    model.fit(Xf[tr], y[tr]); p = model.predict_proba(Xf[te])[:, 1]
    return roc_auc_score(y[te], p), average_precision_score(y[te], p), y[te].mean()
def cv(splits, model):
    r = np.array([score(a, b, model) for a, b in splits]); return r.mean(0)
mk = lambda: RF(100, min_samples_leaf=1, random_state=0)
rows = {}
rows['random stays'] = cv(StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y), mk())
rows['by patient'] = cv(GroupKFold(5).split(Xf, y, pid), mk())
# train on the first two years, test on the last year, keeping the patients apart
test_p = np.unique(pid)[np.unique(pid) % 5 == 0]                 # a fifth of the patients held out
tr = np.where(~np.isin(pid, test_p) & (year <= 2.0))[0]; te = np.where(np.isin(pid, test_p) & (year > 2.0))[0]
rows['by patient and time'] = np.array(score(tr, te, mk()))
rows['logistic, by patient'] = cv(GroupKFold(5).split(Xf, y, pid), LogisticRegression(max_iter=1000))
print("%-24s %8s %8s %12s" % ("protocol", "AUC", "AP", "base rate"))
for k, v in rows.items(): print("%-24s %8.3f %8.3f %12.4f" % (k, *v))
print("test part of the time split: %d stays, %d returns" % (len(te), y[te].sum()))
print("patients with at least two stays: %.2f; stays whose patient also has another stay in a different fold of the random split: ~%.2f" %
      ((n_st >= 2).mean(), (n_st[pid] >= 2).mean()))
print("returns: share of returning stays whose patient has more than one return: %.2f" %
      (np.mean([(y[pid == p].sum() > 1) for p in np.unique(pid[y == 1])])))

fig, axs = plt.subplots(1, 2, figsize=(4.9, 1.75), dpi=240)
names = ['random stays', 'by patient', 'by patient and time']; lab = ['random\nstays', 'by\npatient', 'patient\nand time']
cols = [RED, GREEN, GREEN]
for ax, j, t in ((axs[0], 0, 'AUC'), (axs[1], 1, 'average precision')):
    ax.bar(range(3), [rows[n][j] for n in names], color=cols, edgecolor=NAVY, lw=0.5, width=0.62)
    for i, n in enumerate(names): ax.text(i, rows[n][j] + 0.012, '%.2f' % rows[n][j], ha='center', fontsize=8, color=NAVY)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=8); ax.set_title(t, fontsize=8.6, pad=3)
    ax.tick_params(labelsize=8); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_axisbelow(True)
axs[0].set_ylim(0.5, 1.0); axs[0].axhline(0.5, color=GREY, lw=0.6)
axs[1].set_ylim(0, 0.45); axs[1].axhline(y.mean(), color=NAVY, lw=0.8, ls=':');
fig.subplots_adjust(wspace=0.3, left=0.07, right=0.99, top=0.88, bottom=0.2)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s59c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s59c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

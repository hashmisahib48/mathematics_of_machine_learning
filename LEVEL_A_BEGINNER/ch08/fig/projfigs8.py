"""Sidebar figures for Problems 8-2, 8-3 and 8-4 (real data from scikit-learn)."""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
P = lambda n: os.path.join(HERE, 'png', n)
W = 1.75
def fin(fig, n): fig.tight_layout(pad=0.3, h_pad=0.7); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6.5, length=2.2)
B = load_breast_cancer(); X, y = B.data, B.target; nm = list(B.feature_names)
# P8-2 the redundant measurements
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.0), dpi=300)
for ax, j in ((a, nm.index('mean area')), (b, nm.index('mean texture'))):
    for k, col, lab in ((0, RED, 'malignant'), (1, BLUE, 'benign')):
        ax.plot(X[y == k, 0], X[y == k, j], 'o', ms=1.8, color=col, alpha=0.55, mew=0, label=lab)
sty(a, '', 'mean area'); a.set_xticklabels([]); sty(b, 'mean radius', 'mean texture'); a.legend(fontsize=6.3, loc='upper left', handlelength=0.8, borderaxespad=0.1)
fin(fig, 'pj82.png')
# P8-3 learning curves
sizes = (10, 20, 40, 80, 160, 320); nb, lr = [], []
for n in sizes:
    r = {'nb': [], 'lr': []}
    for tr, te in StratifiedShuffleSplit(n_splits=50, train_size=n, test_size=200, random_state=0).split(X, y):
        r['nb'].append(GaussianNB().fit(X[tr], y[tr]).score(X[te], y[te]))
        r['lr'].append(make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr]).score(X[te], y[te]))
    nb.append(np.mean(r['nb'])); lr.append(np.mean(r['lr']))
fig, ax = plt.subplots(figsize=(W, 1.5), dpi=300)
ax.plot(sizes, nb, 'o-', ms=2.6, lw=1.0, color=BLUE, label='naive Bayes'); ax.plot(sizes, lr, 's-', ms=2.6, lw=1.0, color=RED, label='logistic regression')
ax.set_xscale('log'); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes]); ax.minorticks_off()
sty(ax, 'training size $n$', 'test accuracy'); ax.legend(fontsize=6.3, loc='lower right', handlelength=1.2, borderaxespad=0.1)
fin(fig, 'pj83.png')
# P8-4 iris petal length: the data and the log joints
I = load_iris(); xl = I.data[:, 2]; yt = I.target; g = GaussianNB().fit(xl[:, None], yt); cols = [BLUE, OR, RED]
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.2), dpi=300)
for k, lab in enumerate(['setosa', 'versicolor', 'virginica']):
    a.plot(xl[yt == k], I.data[yt == k, 3], 'o', ms=2.0, color=cols[k], alpha=0.7, mew=0, label=lab)
sty(a, 'petal length (cm)', 'petal width (cm)'); a.legend(fontsize=6.3, loc='upper left', handletextpad=0.1, borderaxespad=0.1)
t = np.linspace(0.5, 7.5, 400)
for k in range(3):
    m, v = g.theta_[k, 0], g.var_[k, 0]; b.plot(t, math.log(1 / 3) - 0.5 * np.log(2 * np.pi * v) - (t - m) ** 2 / (2 * v), color=cols[k], lw=1.0)
b.set_ylim(-14, 2); sty(b, 'petal length (cm)', 'log joint $L_c$')
fin(fig, 'pj84.png')
print('ok', [round(v, 4) for v in nb], [round(v, 4) for v in lr])

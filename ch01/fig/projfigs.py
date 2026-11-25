"""Sidebar figures for the three case studies of Chapter 1 (real data)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from sklearn.datasets import load_wine, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split as split
from sklearn.tree import DecisionTreeRegressor
def two(h): return plt.subplots(2, 1, figsize=(1.75, h), dpi=300)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7, length=2.2)
def fin(fig, n): fig.tight_layout(pad=0.25, h_pad=0.7); fig.savefig(os.path.join(HERE, 'png', n), dpi=300, facecolor='white'); plt.close(fig)
# ---- wines: two features before and after division by the range
W = load_wine(); X, y = W.data, W.target; nm = list(W.feature_names)
P = X[:, nm.index('proline')]; C = X[:, nm.index('color_intensity')]
fig, (a, b) = two(2.55); cols = [BLUE, OR, RED]; mk = ['o', 's', '^']
for k in range(3):
    a.plot(C[y == k], P[y == k], mk[k], ms=2.6, color=cols[k], alpha=0.85, mew=0, label=f'class {k}')
    b.plot((C[y == k] - C.min()) / (C.max() - C.min()), (P[y == k] - P.min()) / (P.max() - P.min()), mk[k], ms=2.6, color=cols[k], alpha=0.85, mew=0)
a.legend(fontsize=6.5, frameon=False, loc='upper right', handletextpad=0.1, borderaxespad=0.1)
sty(a, 'color intensity', 'proline'); sty(b, 'color intensity, scaled', 'proline, scaled'); b.set_aspect('equal')
fin(fig, 'pj1.png')
# ---- tumors: scatter and histogram
B = load_breast_cancer(); X, y = B.data, B.target; nb = list(B.feature_names)
fig, (a, b) = two(2.45); cols = {0: RED, 1: BLUE}; lab = {0: 'malignant', 1: 'benign'}
r, t, cp = X[:, nb.index('mean radius')], X[:, nb.index('mean texture')], X[:, nb.index('mean concave points')]
for k in (1, 0): a.plot(r[y == k], t[y == k], 'o', ms=2.0, color=cols[k], alpha=0.6, mew=0, label=lab[k])
a.legend(fontsize=6.5, frameon=False, loc='upper right', handletextpad=0.1, borderaxespad=0.1); sty(a, 'mean radius', 'mean texture')
bins = np.linspace(0, cp.max(), 25)
for k in (1, 0): b.hist(cp[y == k], bins=bins, color=cols[k], alpha=0.65, label=lab[k])
sty(b, 'mean concave points', 'tumors')
fin(fig, 'pj2.png')
# ---- diabetes: s5 against progression with the depth-1 tree, and the histogram of the target
D = load_diabetes(); X, y = D.data, D.target; a_, b_, c_, d_ = split(X, y, test_size=0.25, random_state=0)
fig, (a, b) = two(2.45)
j = list(D.feature_names).index('s5'); t1 = DecisionTreeRegressor(max_depth=1, random_state=0).fit(a_, c_)
a.plot(a_[:, j], c_, 'o', ms=1.8, color=BLUE, alpha=0.6, mew=0, label='training')
a.plot(b_[:, j], d_, 's', ms=2.0, color=OR, alpha=0.9, mew=0, label='test')
xs = np.array([X[:, j].min(), t1.tree_.threshold[0], X[:, j].max()])
a.plot([xs[0], xs[1]], [t1.tree_.value[1, 0, 0]] * 2, color=GREEN, lw=1.2); a.plot([xs[1], xs[2]], [t1.tree_.value[2, 0, 0]] * 2, color=GREEN, lw=1.2)
a.legend(fontsize=6.5, frameon=False, loc='upper left', handletextpad=0.1, borderaxespad=0.1); sty(a, 's5 (standardized)', 'progression')
b.hist(y, bins=20, color=BLUE, alpha=0.75); sty(b, 'progression', 'patients')
fin(fig, 'pj3.png')
# ---- data tables for the hand calculations (drawn as pictures for the sidebar)
from matplotlib.patches import Rectangle
def tab(name, head, rows, cw, rh=0.19, hh=0.19, fs=7.2):
    W = 1.75; H = hh + rh * len(rows) + 0.02
    fig = plt.figure(figsize=(W, H), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    xs = np.concatenate([[0], np.cumsum(cw)]) * W / sum(cw)
    for j, h in enumerate(head):
        ax.add_patch(Rectangle((xs[j], 0), xs[j + 1] - xs[j], hh, fc=NAVY, ec='white', lw=0.5))
        ax.text((xs[j] + xs[j + 1]) / 2, hh / 2, h, color='white', ha='center', va='center', fontsize=fs, weight='bold', family='Liberation Serif', linespacing=1.12)
    for i, rw in enumerate(rows):
        for j, v in enumerate(rw):
            ax.add_patch(Rectangle((xs[j], hh + i * rh), xs[j + 1] - xs[j], rh, fc=LB if i % 2 == 0 else 'white', ec='#B8C4D6', lw=0.4))
            ax.text((xs[j] + xs[j + 1]) / 2, hh + (i + 0.5) * rh, v, ha='center', va='center', fontsize=fs, family='Liberation Serif', color='#222222')
    fig.savefig(os.path.join(HERE, 'png', name), dpi=300, facecolor='white'); plt.close(fig)
tab('pt1.png', ['Row', 'Proline', 'Color', 'Class'], [['127', '466', '3.00', '1'], ['146', '415', '4.90', '2'], ['37', '1105', '4.25', '0'], ['41', '1035', '4.28', '0'], ['141', '780', '5.60', '2'], ['112', '607', '3.80', '1'], ['105', '315', '2.70', '?']], [0.9, 1.1, 1.0, 0.9])
tab('pt2.png', ['', 'Predicted\nmalignant', 'Predicted\nbenign'], [['Actual malignant', '47', '6'], ['Actual benign', '6', '84']], [1.25, 0.9, 0.8], hh=0.36)
tab('pt3.png', ['Patient', 's5', 'Progression'], [['1', '0.0037', '321'], ['2', '0.0702', '215'], ['3', '-0.0181', '127'], ['4', '-0.0385', '64'], ['5', '-0.0345', '175'], ['6', '0.0617', '275']], [0.8, 1.0, 1.1])
print('ok')

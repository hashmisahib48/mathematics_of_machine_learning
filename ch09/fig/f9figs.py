import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.patches import FancyBboxPatch
import statsmodels.api as sm
from sklearn.datasets import load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.25); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5, labelpad=1.5); ax.set_ylabel(yl, fontsize=7.5, labelpad=1.5); ax.tick_params(labelsize=7, length=2.2, pad=1.5)
# 9.2-1 impurity
p = np.linspace(0, 1, 300); fig, ax = plt.subplots(figsize=(1.75, 1.6), dpi=300)
with np.errstate(divide='ignore', invalid='ignore'): H = np.nan_to_num(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))
ax.plot(p, H / 2, color=BLUE, lw=1.1, label='entropy / 2'); ax.plot(p, 2 * p * (1 - p), color=RED, lw=1.1, label='Gini')
ax.plot(p, np.minimum(p, 1 - p), color=GREY, lw=0.9, ls='--', label='error rate')
ax.legend(fontsize=7, frameon=False, loc='upper right', handlelength=1.4, borderaxespad=0.1, labelspacing=0.25); ax.set_ylim(0, 0.8); ax.set_yticks([0, 0.2, 0.4]); sty(ax, 'fraction $p$ of class 1', 'impurity'); fin(fig, 'f91.png')
# 9.4-1 tree + regions
X = np.array([[1, 4], [2, 7], [3, 2], [5, 6], [6, 2], [6, 8], [7, 5], [8, 4]]); y = np.array([0, 0, 0, 1, 0, 1, 1, 0])
fig, (a, b) = plt.subplots(2, 1, figsize=(1.75, 3.45), dpi=300, gridspec_kw={'height_ratios': [1, 1.2]})
a.axis('off'); a.set_xlim(0, 10); a.set_ylim(0, 10)
def box(ax, x, y, t, leaf=None):
    fc = '#FFFFFF' if leaf is None else ('#DCE8F8' if leaf == 0 else '#FDEBD3')
    ax.add_patch(FancyBboxPatch((x - 1.5, y - 1.0), 3.0, 2.0, boxstyle='round,pad=0.1', fc=fc, ec=NAVY, lw=0.7, zorder=3)); ax.text(x, y, t, ha='center', va='center', fontsize=7, zorder=4, linespacing=1.05)
for (x0, y0, x1, y1) in [(4.8, 7.9, 2.0, 6.2), (4.8, 7.9, 7.0, 6.2), (7.0, 4.0, 5.0, 2.3), (7.0, 4.0, 8.4, 2.3)]:
    a.plot([x0, x1], [y0, y1], color=NAVY, lw=0.6, zorder=1)
box(a, 4.8, 8.9, '$x_2\\leq 4.5$\n8: (5, 3)'); box(a, 1.9, 5.2, 'class 0\n4: (4, 0)', 0); box(a, 7.0, 5.0, '$x_1\\leq 3.5$\n4: (1, 3)')
box(a, 5.0, 1.3, 'class 0\n1: (1, 0)', 0); box(a, 8.4, 1.3, 'class 1\n3: (0, 3)', 1)
a.text(2.6, 7.2, 'yes', fontsize=7, color=GREY); a.text(6.4, 7.2, 'no', fontsize=7, color=GREY)
a.text(5.2, 3.2, 'yes', fontsize=7, color=GREY, ha='right'); a.text(7.9, 3.2, 'no', fontsize=7, color=GREY)
b.fill_between([0, 9], 0, 4.5, color='#DCE8F8'); b.fill_between([0, 3.5], 4.5, 9, color='#DCE8F8'); b.fill_between([3.5, 9], 4.5, 9, color='#FDEBD3')
b.axhline(4.5, color=NAVY, lw=0.8); b.plot([3.5, 3.5], [4.5, 9], color=NAVY, lw=0.8)
off = {'A': (0.0, -1.05), 'H': (0.0, -1.05), 'G': (0.25, 0.4)}
for (u, v), k, nm in zip(X, y, 'ABCDEFGH'):
    b.plot(u, v, 'o' if k == 0 else 's', ms=3.5, color=BLUE if k == 0 else OR); dx, dy = off.get(nm, (0.25, 0.35)); b.text(u + dx, v + dy, nm, fontsize=7)
b.set_xlim(0, 9); b.set_ylim(0, 9); sty(b, '$x_1$', '$x_2$'); b.set_xticks([0, 2, 4, 6, 8]); fin(fig, 'f92.png')
# 9.7-1 diabetes
Xd, yd = load_diabetes(return_X_y=True); a_, b_, c_, d_ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
L = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 120]; tr, te = [], []
for l in L: m = DecisionTreeRegressor(min_samples_leaf=l, random_state=0).fit(a_, c_); tr.append(m.score(a_, c_)); te.append(m.score(b_, d_))
fig, ax = plt.subplots(figsize=(1.75, 1.75), dpi=300)
ax.plot(L, tr, 'o-', ms=2.2, color=BLUE, lw=0.9, label='training'); ax.plot(L, te, 's-', ms=2.2, color=RED, lw=0.9, label='test')
ax.axhline(0.359, color=GREEN, lw=0.8, ls='--'); ax.text(1.1, 0.43, 'linear regression', fontsize=7, color=GREEN)
ax.set_xscale('log'); ax.axhline(0, color='k', lw=0.4); ax.set_ylim(-0.3, 1.05); ax.legend(fontsize=7, frameon=False, loc='upper right', handlelength=1.4, borderaxespad=0.1, labelspacing=0.25); sty(ax, 'smallest leaf (observations)', '$R^2$'); fin(fig, 'f93.png')
# P9-1 wine
Wn = load_wine(); fig, ax = plt.subplots(figsize=(1.75, 1.6), dpi=300)
for k, (mk, col) in enumerate([('o', BLUE), ('s', OR), ('^', GREEN)]): ax.plot(Wn.data[Wn.target == k, 6], Wn.data[Wn.target == k, 12], mk, ms=2.4, color=col, alpha=0.8, label=f'cultivar {k}', mew=0)
ax.legend(fontsize=7, frameon=False, loc='upper left', handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2); ax.set_ylim(200, 1900); sty(ax, 'flavanoids', 'proline'); fin(fig, 'pj91.png')
# P9-2 engel
e = sm.datasets.engel.load_pandas().data; fig, ax = plt.subplots(figsize=(1.75, 1.6), dpi=300)
ax.plot(e.income, e.foodexp, 'o', ms=1.8, color=BLUE, alpha=0.7); wl, bl = np.polyfit(e.income, e.foodexp, 1); xl = np.array([e.income.min(), e.income.max()]); ax.plot(xl, wl * xl + bl, color=GREEN, lw=0.9)
ax.set_xticks([1000, 2000, 3000, 4000]); sty(ax, 'income (francs per year)', 'food (francs per year)'); fin(fig, 'pj92.png')
# P9-3 breast cancer
Bc = load_breast_cancer(); fig, ax = plt.subplots(figsize=(1.75, 1.6), dpi=300)
for k, (mk, col, nm) in enumerate([('o', RED, 'malignant'), ('s', BLUE, 'benign')]): ax.plot(Bc.data[Bc.target == k, 22], Bc.data[Bc.target == k, 27], mk, ms=2.0, color=col, alpha=0.7, label=nm, mew=0)
ax.legend(fontsize=7, frameon=False, loc='lower right', handletextpad=0.2, borderaxespad=0.1, labelspacing=0.2); sty(ax, 'worst perimeter', 'worst concave points'); fin(fig, 'pj93.png')
print('ok')

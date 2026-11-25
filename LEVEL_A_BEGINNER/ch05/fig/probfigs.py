"""Data figures for the five problems of Chapter 5 (sidebar, 1.75 in wide, true-size fonts)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
import pandas as pd
from sklearn.datasets import load_diabetes
D = os.path.join(HERE, '..', 'data'); PNG = lambda n: os.path.join(HERE, 'png', n)
PURPLE = '#7B3FA0'
def sty(ax, xl, yl):
    ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7, length=2.2)
def fin(fig, name): fig.tight_layout(pad=0.25, h_pad=0.7); fig.savefig(PNG(name), dpi=300, facecolor='white'); plt.close(fig)
# 5-1 stack loss: loss against air flow with the line, and the range of each operating condition
d = pd.read_csv(os.path.join(D, 'stackloss.csv')); cols = ['AIRFLOW', 'WATERTEMP', 'ACIDCONC']
fig, (a, b) = plt.subplots(2, 1, figsize=(1.75, 1.6), dpi=300, gridspec_kw={'height_ratios': [1.25, 1]})
w, c = np.polyfit(d.AIRFLOW, d.STACKLOSS, 1); xx = np.array([48, 82])
a.plot(d.AIRFLOW, d.STACKLOSS, 'o', ms=3, color=BLUE); a.plot(xx, w * xx + c, color=GREEN, lw=1.1); sty(a, 'air flow', 'stack loss')
for k, (nm, col) in enumerate(zip(['air flow', 'water temp.', 'acid conc.'], cols)):
    b.plot([d[col].min(), d[col].max()], [k, k], '-', color=OR, lw=3, solid_capstyle='butt')
    b.text(d[col].max() + 2, k, nm, fontsize=6.5, va='center', color=NAVY)
b.set_yticks([]); b.set_ylim(-0.7, 2.7); b.set_xlim(0, 125); sty(b, 'range of the raw values', '')
print('5-1 ranges', {col: (d[col].min(), d[col].max(), round(d[col].mean(), 1)) for col in cols})
fin(fig, 'pj5_1.png')
# 5-2 diabetes: the target, and the target against the body mass index
X, y = load_diabetes(return_X_y=True); bmi = X[:, 2]
fig, (a, b) = plt.subplots(2, 1, figsize=(1.75, 1.6), dpi=300)
a.hist(y, bins=16, color=BLUE, edgecolor='white', lw=0.4); sty(a, 'disease progression', 'patients')
b.plot(bmi, y, 'o', ms=1.8, color=BLUE, alpha=0.6); sty(b, 'standardized body mass index', 'progression')
fin(fig, 'pj5_2.png')
# 5-3 Longley: three features that rise together, and the employment
d = pd.read_csv(os.path.join(D, 'longley.csv')); z = lambda v: (v - v.mean()) / v.std()
fig, (a, b) = plt.subplots(2, 1, figsize=(1.75, 2.1), dpi=300)
for col, cc, nm in [('GNP', BLUE, 'GNP'), ('POP', RED, 'population'), ('GNPDEFL', GREEN, 'deflator')]: a.plot(d.YEAR, z(d[col]), '-o', ms=2, lw=0.8, color=cc, label=nm)
a.set_ylim(-1.9, 2.9); a.legend(fontsize=5.6, frameon=False, loc='upper left', ncol=3, columnspacing=0.6, handlelength=1.0, handletextpad=0.3); sty(a, 'year', 'standardized')
b.plot(d.YEAR, d.TOTEMP / 1000, 'o-', ms=2.5, lw=0.8, color=PURPLE); sty(b, 'year', 'employed (millions)')
print('5-3 corr GNP POP YEAR', np.round(np.corrcoef(d[['GNP', 'POP', 'YEAR']].values.T), 3).tolist())
fin(fig, 'pj5_3.png')
# 5-4 Engel: logarithms of income and food expenditure
d = pd.read_csv(os.path.join(D, 'engel.csv')); lx, ly = np.log(d.income.values), np.log(d.foodexp.values)
fig, a = plt.subplots(figsize=(1.75, 1.5), dpi=300)
w, c = np.polyfit(lx, ly, 1); xx = np.array([lx.min(), lx.max()])
a.plot(lx, ly, 'o', ms=1.8, color=BLUE, alpha=0.7); a.plot(xx, w * xx + c, color=GREEN, lw=1.1); sty(a, 'ln income', 'ln food expenditure')
fin(fig, 'pj5_4.png')
# 5-5 mini-batches: one epoch of twelve observations cut into groups of m (schematic)
from matplotlib.patches import Rectangle
fig, a = plt.subplots(figsize=(1.75, 1.25), dpi=300); rng = np.random.default_rng(3); order = rng.permutation(12) + 1
cols_ = [BLUE, GREEN, OR, RED, PURPLE, '#1A8A8A']
for row, (m, ttl) in enumerate([(1, 'm = 1'), (4, 'm = 4'), (12, 'm = 12')]):
    y0 = 2.8 - 1.4 * row
    for k, o in enumerate(order):
        g = k // m; a.add_patch(Rectangle((k, y0 + 0.08), 0.86, 0.5, fc=cols_[g % 6], ec='none', alpha=0.85))
        a.text(k + 0.43, y0 + 0.33, str(o), fontsize=5.6, color='white', ha='center', va='center')
    a.text(0, y0 + 0.78, ttl, fontsize=7, color=NAVY, va='bottom')
a.set_xlim(-0.1, 12.1); a.set_ylim(-0.1, 4.2); a.axis('off')
fin(fig, 'pj5_5.png')

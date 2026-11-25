"""Sidebar figures for the three projects of Chapter 3 (real data)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib')); sys.path.insert(0, os.path.join(HERE, '..', 'code'))
from figstyle import *
import pandas as pd
from sklearn.linear_model import LinearRegression
D = os.path.join(HERE, '..', 'data')
def two(name, h=2.7):
    fig, axs = plt.subplots(2, 1, figsize=(2.05, h), dpi=300); return fig, axs
def fin(fig, name): fig.tight_layout(pad=0.25, h_pad=0.6); fig.savefig(os.path.join(HERE, 'png', name), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl):
    ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6.3, length=2.2)
# ---- stack loss
d = pd.read_csv(os.path.join(D, 'stackloss.csv')); X = d[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; y = d.STACKLOSS.values
fig, (a, b) = two('p1')
w, c = np.polyfit(X[:, 0], y, 1); xx = np.array([48, 82])
a.plot(X[:, 0], y, 'o', ms=3, color=BLUE); a.plot(xx, w * xx + c, color=GREEN, lw=1.1); sty(a, 'air flow', 'stack loss')
m = LinearRegression().fit(X, y); f = m.predict(X); e = y - f
b.axhline(0, color=GREEN, lw=0.9); b.plot(f, e, 'o', ms=3, color=RED)
for k in [20, 3, 2]: b.annotate(str(k + 1), (f[k], e[k]), xytext=(3, 0), textcoords='offset points', fontsize=6, color=RED, va='center')
sty(b, 'fitted stack loss (three features)', 'error')
fin(fig, 'pj1.png')
# ---- Longley
d = pd.read_csv(os.path.join(D, 'longley.csv'))
fig, (a, b) = two('p2')
cols = ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']; m = LinearRegression().fit(d[cols].values, d.TOTEMP.values)
a.plot(d.YEAR, d.TOTEMP / 1000, 'o', ms=3, color=BLUE, label='measured'); a.plot(d.YEAR, m.predict(d[cols].values) / 1000, '-', color=GREEN, lw=1.0, label='fitted')
a.legend(fontsize=5.8, frameon=False, loc='upper left'); sty(a, 'year', 'employed (millions)')
b.plot(d.GNP / 1000, d.POP / 1000, 'o', ms=3, color=PURPLE if 'PURPLE' in dir() else '#7B3FA0'); sty(b, 'GNP (billions of dollars)', 'population (millions)')
fin(fig, 'pj2.png')
# ---- Engel
d = pd.read_csv(os.path.join(D, 'engel.csv')); x = d.income.values; y = d.foodexp.values
fig, (a, b) = two('p3')
w, c = np.polyfit(x, y, 1); xx = np.array([300, 5100])
a.plot(x, y, 'o', ms=1.8, color=BLUE, alpha=0.7); a.plot(xx, w * xx + c, color=GREEN, lw=1.1); sty(a, 'income (francs per year)', 'food (francs per year)')
lw, lc = np.polyfit(np.log(x), np.log(y), 1); lx = np.log(xx)
b.plot(np.log(x), np.log(y), 'o', ms=1.8, color=BLUE, alpha=0.7); b.plot(lx, lw * lx + lc, color=GREEN, lw=1.1); sty(b, 'ln income', 'ln food')
fin(fig, 'pj3.png')
print('ok')
# ---- Linnerud
d = pd.read_csv(os.path.join(D, 'linnerud.csv'))
fig, (a, b) = two('p4')
w, c = np.polyfit(d.Waist, d.Situps, 1); xx = np.array([30, 47])
a.plot(d.Waist, d.Situps, 'o', ms=3, color=BLUE); a.plot(xx, w * xx + c, color=GREEN, lw=1.1); sty(a, 'waist (in)', 'sit-ups')
b.plot(d.Weight, d.Waist, 'o', ms=3, color='#7B3FA0'); sty(b, 'weight (lb)', 'waist (in)')
fin(fig, 'pj4.png')
print('ok4')

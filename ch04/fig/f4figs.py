import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import load_diabetes
import statsmodels.api as sm, pandas as pd
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7, length=2.2)
# 4.1-1 straight line and parabola
x = np.array([-2, -1, 0, 1, 2.]); y = np.array([4.1, 1.2, 0.1, 0.9, 3.9]); t = np.linspace(-2.3, 2.3, 200)
fig, ax = plt.subplots(figsize=(2.05, 1.75), dpi=300)
ax.plot(t, np.polyval(np.polyfit(x, y, 1), t), color=OR, lw=1.1, label='straight line')
ax.plot(t, 0.9786 * t ** 2 - 0.07 * t + 0.0829, color=GREEN, lw=1.2, label='quadratic')
ax.plot(x, y, 'o', ms=3.6, color=BLUE, zorder=4); ax.legend(fontsize=5.8, frameon=False, loc='upper center'); ax.set_ylim(-0.8, 6)
sty(ax, '$x$', '$y$'); fin(fig, 'f41.png')
# 4.2-1 degree 9, LS and ridge
rng = np.random.default_rng(0); x = np.sort(rng.uniform(0, 4, 12)); y = np.sin(1.3 * x) + 0.3 * x + rng.normal(0, 0.25, 12)
t = np.linspace(x.min(), x.max(), 400)
fig, ax = plt.subplots(figsize=(2.05, 1.75), dpi=300)
ax.plot(t, np.sin(1.3 * t) + 0.3 * t, color=GREY, lw=0.8, ls='--', label='true curve')
for est, c, lab in [(LinearRegression(), RED, 'least squares'), (Ridge(0.001), GREEN, r'ridge, $\alpha=0.001$')]:
    m = make_pipeline(PolynomialFeatures(9, include_bias=False), StandardScaler(), est).fit(x[:, None], y)
    ax.plot(t, m.predict(t[:, None]), color=c, lw=1.1, label=lab)
ax.plot(x, y, 'o', ms=3.2, color=NAVY, zorder=4); ax.set_ylim(-1.2, 2.8); ax.legend(fontsize=5.4, frameon=False, loc='upper right')
sty(ax, '$x$', '$y$'); fin(fig, 'f42.png')
# 4.3-1 ridge path on diabetes
X, y = load_diabetes(return_X_y=True); Z = StandardScaler().fit_transform(X); al = np.logspace(-2, 4, 60)
W = np.array([Ridge(alpha=a).fit(Z, y).coef_ for a in al])
fig, ax = plt.subplots(figsize=(2.05, 2.1), dpi=300)
cols = [BLUE, OR, GREEN, RED, PUR, TEAL, MAG, NAVY, '#8A6D12', GREY]
for j, nm in enumerate(load_diabetes().feature_names): ax.plot(al, W[:, j], color=cols[j], lw=1.0, label=nm)
ax.set_xscale('log'); ax.axhline(0, color='k', lw=0.4); sty(ax, r'penalty $\alpha$', 'weight $w_j$')
ax.legend(fontsize=6.3, ncol=5, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=0.9, columnspacing=0.6, handletextpad=0.3, borderaxespad=0.2)
fin(fig, 'f43.png')
# P4-1 Longley ridge path
d = sm.datasets.longley.load_pandas().data; LAB6 = ['deflator', 'GNP', 'unemployed', 'armed forces', 'population', 'year']; cols6 = ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']
Z = StandardScaler().fit_transform(d[cols6].values); y = d.TOTEMP.values; al = np.logspace(-3, 2, 60)
W = np.array([Ridge(alpha=a).fit(Z, y).coef_ for a in al])
fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
for j in range(6): ax.plot(al, W[:, j] / 1000, color=cols[j], lw=0.9, label=LAB6[j])
ax.set_xscale('log'); ax.axhline(0, color='k', lw=0.4); ax.legend(fontsize=6.3, frameon=False, ncol=2, loc='upper right', handlelength=1.2, columnspacing=0.8); sty(ax, r'penalty $\alpha$', 'weight (thousands)')
fin(fig, 'pj41.png')
# P4-3 Engel data
e = pd.read_csv(os.path.join(HERE, '..', 'data', 'engel.csv')); inc = e.income.values / 1000; fd = e.foodexp.values
fig, (a1, a2) = plt.subplots(2, 1, figsize=(2.05, 2.5), dpi=300)
w1, c1 = np.polyfit(inc, fd, 1); xx = np.array([0.3, 5.1])
a1.plot(inc, fd, 'o', ms=1.8, color=BLUE, alpha=0.7); a1.plot(xx, w1 * xx + c1, color=GREEN, lw=1.0); sty(a1, 'income (thousand francs)', 'food (francs)')
a2.hist(inc, bins=np.arange(0, 5.5, 0.5), color=BLUE, ec='white', lw=0.4); sty(a2, 'income (thousand francs)', 'households')
fig.tight_layout(pad=0.25, h_pad=0.7); fig.savefig(P('pj43.png'), dpi=300, facecolor='white'); plt.close(fig)
# P4-4 diabetes data
db = load_diabetes(); Xd, yd = db.data, db.target
fig, ax = plt.subplots(figsize=(2.05, 1.8), dpi=300)
bm = Xd[:, 2]; w2, c2 = np.polyfit(bm, yd, 1); xs = np.array([bm.min(), bm.max()])
ax.plot(bm, yd, 'o', ms=1.8, color=BLUE, alpha=0.6); ax.plot(xs, w2 * xs + c2, color=GREEN, lw=1.0); sty(ax, 'body mass index (standardized)', 'progression')
fin(fig, 'pj44.png'); print('ok')

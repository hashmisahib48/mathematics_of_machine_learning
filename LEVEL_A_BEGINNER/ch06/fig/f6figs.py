import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, load_diabetes, load_digits
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.5); ax.set_ylabel(yl, fontsize=7.5); ax.tick_params(labelsize=7, length=2.2)
# 6.1-1 worked example
X = np.array([[1, 1], [2, 2], [1, 3], [4, 4], [5, 2], [3.5, 1.]]); c = [0, 0, 0, 1, 1, 1]; q = np.array([3, 2.5])
fig, ax = plt.subplots(figsize=(1.75, 1.75), dpi=300)
for (a, b), k, nm in zip(X, c, 'ABCDEF'):
    ax.plot(a, b, 'o' if k == 0 else 's', ms=5, color=BLUE if k == 0 else OR, zorder=4); ax.text(a + (-0.42 if nm == 'C' else 0.14), b + (0.1 if nm != 'C' else 0.12), nm, fontsize=7.5)
ax.plot(*q, '*', ms=8, color=RED, zorder=5); ax.text(q[0] + 0.14, q[1] - 0.42, 'z', fontsize=8, color=RED, style='italic')
t = np.linspace(0, 2 * np.pi, 200)
for r, ls, lab, dy in [(1.118, '-', 'k = 1', 0.08), (1.803, '--', 'k = 3', 0.08)]:
    ax.plot(q[0] + r * np.cos(t), q[1] + r * np.sin(t), color=GREY, lw=0.7, ls=ls); ax.text(q[0], q[1] + r + dy, lab, fontsize=6.5, color=GREY, ha='center')
ax.set_aspect('equal'); ax.set_xlim(0.3, 5.7); ax.set_ylim(0.3, 4.9); sty(ax, '$x_1$', '$x_2$'); fin(fig, 'f61.png')
# 6.2-1 decision regions iris
I = load_iris(); Xi = I.data[:, 2:]; yi = I.target
cm = ListedColormap(['#DCE8F8', '#FDEBD3', '#F7DAD6'])
fig, axs = plt.subplots(2, 1, figsize=(1.75, 3.2), dpi=300)
a, b = np.meshgrid(np.linspace(0.8, 7.1, 300), np.linspace(0, 2.7, 300))
for ax, k in zip(axs, (1, 25)):
    m = KNeighborsClassifier(k).fit(Xi, yi); Zp = m.predict(np.c_[a.ravel(), b.ravel()]).reshape(a.shape)
    ax.contourf(a, b, Zp, levels=[-0.5, 0.5, 1.5, 2.5], cmap=cm)
    for j, (col, mk) in enumerate(zip([BLUE, OR, RED], ['o', 's', '^'])): ax.plot(Xi[yi == j, 0], Xi[yi == j, 1], mk, ms=1.8, color=col)
    ax.set_title(f'k = {k}', fontsize=8, color=NAVY); sty(ax, 'petal length (cm)', 'petal width (cm)')
fin(fig, 'f62.png')
# 6.7-1 diabetes R2 vs k
Xd, yd = load_diabetes(return_X_y=True); a_, b_, c_, d_ = train_test_split(Xd, yd, test_size=0.25, random_state=0)
ks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60, 80, 120]; tr, te = [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsRegressor(k)).fit(a_, c_); tr.append(m.score(a_, c_)); te.append(m.score(b_, d_))
fig, ax = plt.subplots(figsize=(1.75, 1.75), dpi=300)
ax.plot(ks, tr, 'o-', ms=2.2, color=BLUE, lw=0.9, label='training'); ax.plot(ks, te, 's-', ms=2.2, color=RED, lw=0.9, label='test')
ax.set_xscale('log'); ax.set_ylim(-0.7, 1.05); ax.axhline(0, color='k', lw=0.4); ax.legend(fontsize=7, frameon=False, loc='center right')
sty(ax, 'number of neighbors $k$', '$R^2$'); fin(fig, 'f63.png')
# P6-2 digits
D = load_digits(); fig, axs = plt.subplots(2, 5, figsize=(1.75, 0.82), dpi=300)
for i, ax in enumerate(axs.flat): ax.imshow(D.images[i], cmap='gray_r'); ax.axis('off'); ax.set_title(str(D.target[i]), fontsize=7, pad=1)
fin(fig, 'pj62.png')

# ---- Problem 6-1: accuracy against k, raw and standardized (the values are not printed)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
Xb, yb = load_breast_cancer(return_X_y=True); kk = [1, 3, 5, 11, 21]
raw = [cross_val_score(KNeighborsClassifier(k), Xb, yb, cv=5).mean() for k in kk]
std = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), Xb, yb, cv=5).mean() for k in kk]
fig, ax = plt.subplots(figsize=(1.75, 1.7), dpi=300)
ax.plot(kk, std, 'o-', ms=3, lw=0.9, color=BLUE, label='standardized'); ax.plot(kk, raw, 's-', ms=3, lw=0.9, color=RED, label='raw')
ax.set_ylim(0.88, 1.0); ax.set_xticks(kk); ax.legend(fontsize=7, frameon=False, loc='lower right'); sty(ax, 'number of neighbors $k$', 'accuracy'); fin(fig, 'pj61.png')
# ---- Problem 6-3: stack loss against air flow, with the neighbors of an air flow of 72
import statsmodels.api as sm
sl = sm.datasets.stackloss.load_pandas().data; xa, ya = sl.AIRFLOW.values, sl.STACKLOSS.values
w_, c_ = np.polyfit(xa, ya, 1); xx = np.array([48, 82])
fig, ax = plt.subplots(figsize=(1.75, 1.75), dpi=300)
ax.plot(xa, ya, 'o', ms=3, color=BLUE, label='days'); ax.plot(xx, w_ * xx + c_, color=GREEN, lw=1.0, label='line')
nb = np.argsort(np.abs(xa - 72), kind='stable')[:2]
ax.plot(xa[nb], ya[nb], 'o', ms=6.5, mfc='none', mec=OR, mew=1.2, label='neighbors')
ax.axvline(72, color=GREY, lw=0.7, ls='--'); ax.text(71, 44, '72', fontsize=7, color=GREY, ha='right')
ax.legend(fontsize=6.5, frameon=False, loc='upper left', borderaxespad=0.1, handletextpad=0.3); ax.set_ylim(4, 47)
sty(ax, 'air flow', 'stack loss'); fin(fig, 'pj63.png')
# ---- Problem 6-4: the eight flowers and flower 83 in the petal plane
Xi, yi = load_iris(return_X_y=True); Pi = Xi[:, 2:]; rows = [52, 56, 63, 77, 106, 119, 126, 134]
fig, ax = plt.subplots(figsize=(1.75, 1.75), dpi=300)
off = {52: (-4, -8), 56: (-9, 2), 63: (-9, -3), 77: (-4, 4), 106: (-3, 4), 119: (1, -8), 126: (0, 4), 134: (-4, 4)}
for r_ in rows:
    ve = yi[r_] == 1; ax.plot(*Pi[r_], 'o' if ve else '^', ms=3.4, color=BLUE if ve else RED)
    ax.annotate(str(r_), Pi[r_], xytext=off[r_], textcoords='offset points', fontsize=6.3, color=BLUE if ve else RED)
ax.plot(*Pi[83], '*', ms=8, color=OR, mec=NAVY, mew=0.4); ax.annotate('83', Pi[83], xytext=(3, 1), textcoords='offset points', fontsize=6.3, color=NAVY)
ax.set_xlim(4.35, 5.8); ax.set_ylim(1.25, 1.95); sty(ax, 'petal length (cm)', 'petal width (cm)'); fin(fig, 'pj64.png')
print('ok')

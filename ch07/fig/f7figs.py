import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.tight_layout(pad=0.3); fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7.6); ax.set_ylabel(yl, fontsize=7.6); ax.tick_params(labelsize=7, length=2.2)
s = lambda t: 1 / (1 + np.exp(-t))
# 7.2-1 sigmoid
t = np.linspace(-6, 6, 300); fig, ax = plt.subplots(figsize=(2.05, 1.55), dpi=300)
ax.plot(t, s(t), color=BLUE, lw=1.3); ax.plot([-2, 2], [0, 1], color=OR, lw=0.9, ls='--')
ax.axhline(0.5, color=GREY, lw=0.4); ax.axvline(0, color=GREY, lw=0.4)
ax.text(1.0, 0.12, 'slope 1/4 at $t=0$', fontsize=6.8, color=OR); ax.set_ylim(-0.05, 1.05)
sty(ax, 'score $t$', r'$\sigma(t)$'); fin(fig, 'f71.png')
# 7.2-2 loss per observation
p = np.linspace(0.005, 0.995, 300); fig, ax = plt.subplots(figsize=(2.05, 1.55), dpi=300)
ax.plot(p, -np.log(p), color=BLUE, lw=1.2, label='$y=1$: $-\\ln p$'); ax.plot(p, -np.log(1 - p), color=RED, lw=1.2, label='$y=0$: $-\\ln(1-p)$')
ax.set_ylim(0, 5); ax.legend(fontsize=6.8, frameon=False, loc='upper center'); sty(ax, 'predicted probability $p$', 'cost'); fin(fig, 'f72.png')
# 7.4-1 example
x = np.arange(1, 7); y = np.array([0, 0, 1, 0, 1, 1]); w, b = 1.2140, -4.2491
fig, ax = plt.subplots(figsize=(2.05, 1.55), dpi=300); xx = np.linspace(0, 7, 200)
ax.plot(xx, s(w * xx + b), color=BLUE, lw=1.2); ax.plot(x, y, 'o', ms=3.5, color=OR, zorder=4)
ax.axvline(3.5, color=GREY, lw=0.6, ls='--'); ax.axhline(0.5, color=GREY, lw=0.4); ax.text(3.6, 0.05, 'boundary', fontsize=6.8, color=GREY)
ax.set_ylim(-0.08, 1.08); sty(ax, 'hours of study $x$', 'P(pass)'); fin(fig, 'f73.png')
# 7.7-1 threshold
X, Y = load_breast_cancer(return_X_y=True); a, b_, c, d = train_test_split(X, Y, test_size=0.25, random_state=0, stratify=Y)
m = make_pipeline(StandardScaler(), LogisticRegression()).fit(a, c); pm = m.predict_proba(b_)[:, 0]
th = np.logspace(-2.5, np.log10(0.9), 200); miss = [((d == 0) & (pm < q)).sum() for q in th]; fal = [((d == 1) & (pm >= q)).sum() for q in th]
fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300)
ax.step(th, miss, color=RED, lw=1.1, where='post', label='missed malignant'); ax.step(th, fal, color=BLUE, lw=1.1, where='post', label='false alarms')
ax.set_xscale('log'); ax.axvline(0.5, color=GREY, lw=0.5, ls='--'); ax.legend(fontsize=6.8, frameon=False); sty(ax, 'threshold on P(malignant)', 'test cases'); fin(fig, 'f74.png')
# P7-1 spector
D = sm.datasets.spector.load_pandas().data; mm = LogisticRegression(C=np.inf).fit(D[['GPA', 'TUCE', 'PSI']].values, D.GRADE.values)
fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300); g = np.linspace(2, 4.05, 200); tu = D.TUCE.mean()
for psi, col, lab in [(0, BLUE, 'traditional'), (1, OR, 'PSI')]:
    q = D[D.PSI == psi]; ax.plot(q.GPA, q.GRADE + (0.03 if psi else -0.03), 'o' if psi == 0 else 's', ms=3.0, color=col)
    ax.plot(g, mm.predict_proba(np.c_[g, np.full_like(g, tu), np.full_like(g, psi)])[:, 1], color=col, lw=1.1, label=lab)
ax.legend(fontsize=6.8, frameon=False, loc='center left'); ax.set_ylim(-0.1, 1.1); sty(ax, 'grade point average', 'P(grade improved)'); fin(fig, 'pj71.png')
# 7.7-2 iris probabilities along the petal width
from sklearn.datasets import load_iris
Xi, yi = load_iris(return_X_y=True); mi = make_pipeline(StandardScaler(), LogisticRegression(C=10)).fit(Xi, yi)
pw = np.linspace(0, 2.6, 200); Q = np.tile(Xi.mean(axis=0), (200, 1)); Q[:, 3] = pw; pr = mi.predict_proba(Q)
fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300)
for k, (nm, col) in enumerate([('setosa', BLUE), ('versicolor', OR), ('virginica', RED)]): ax.plot(pw, pr[:, k], color=col, lw=1.2, label=nm)
ax.legend(fontsize=6.8, frameon=False, loc='center left', bbox_to_anchor=(0.2, 0.5)); ax.set_ylim(-0.05, 1.05); sty(ax, 'petal width (cm)', 'probability'); fin(fig, 'f75.png')
# P7-4 imbalanced iris
sel = np.r_[0:100, 140:150]; Xr = Xi[sel][:, 2:]; yr = yi[sel]
fig, ax = plt.subplots(figsize=(2.05, 1.6), dpi=300)
gx, gy = np.meshgrid(np.linspace(2.8, 7.0, 300), np.linspace(0.0, 2.8, 300)); G = np.c_[gx.ravel(), gy.ravel()]
for cw, col, ls in [(None, GREY, '-'), ('balanced', RED, '--')]:
    mq = make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw)).fit(Xr, yr)
    ax.contour(gx, gy, (mq.predict(G) == 2).reshape(gx.shape), [0.5], colors=col, linewidths=1.0, linestyles=ls)
for k, (col, mk) in enumerate([(BLUE, 'o'), (OR, 's'), (RED, '^')]):
    q = Xr[yr == k]; ax.plot(q[:, 0], q[:, 1], mk, ms=2.4, color=col, label=['setosa', 'versicolor', 'virginica (10)'][k])
ax.plot([], [], color=GREY, lw=1.0, label='unweighted'); ax.plot([], [], color=RED, lw=1.0, ls='--', label='balanced')
ax.legend(fontsize=6.0, frameon=False, loc='upper left', handlelength=1.2, borderaxespad=0.1, ncol=2, columnspacing=0.8)
ax.set_xlim(0.8, 7.2); ax.set_ylim(0, 3.4)
sty(ax, 'petal length (cm)', 'petal width (cm)'); fin(fig, 'pj74.png')
print('ok')

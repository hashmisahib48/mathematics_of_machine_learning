# Problem 18C.1: an outlier and the boundary. Writes fig/s18c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import MinCovDet
np.set_printoptions(precision=3, suppress=True)
rg = np.random.default_rng(3)                   # the data of the problem figure
X0 = rg.multivariate_normal([0, 0], [[1, 0.3], [0.3, 0.6]], 40); X1 = rg.multivariate_normal([3, 1.5], [[1, 0.3], [0.3, 0.6]], 40)
Xc = np.r_[X0, X1]; yc = np.r_[np.zeros(40), np.ones(40)].astype(int)
out = np.array([7.5, -3.0]); out2 = np.array([-7.5, 3.0])
ang = lambda w: float(np.degrees(np.arctan2(w[1], w[0])))
bx = lambda w, b, x2=0.75: float(-(w[1] * x2 + b) / w[0])       # boundary position along x1 at x2 = 0.75
mk = {'LDA': lambda: LDA(solver='lsqr'), 'LDA, MinCovDet': lambda: LDA(solver='lsqr', covariance_estimator=MinCovDet(random_state=0)),
      'logistic regression': lambda: LogisticRegression(max_iter=10000)}
print("== the problem's outlier at (7.5, -3), class 0, on the wrong side")
Xo = np.r_[Xc, [out]]; yo = np.r_[yc, [0]]
for nm, (X, y) in {'without': (Xc, yc), 'with': (Xo, yo)}.items():
    m = LDA(solver='lsqr').fit(X, y)
    print(f"{nm:8s} priors {m.priors_}  mean0 {m.means_[0]}  mean1 {m.means_[1]}  Sigma11 {m.covariance_[0, 0]:.3f}  Sigma12 {m.covariance_[0, 1]:.3f}  Sigma22 {m.covariance_[1, 1]:.3f}")
    print(f"         w {m.coef_[0]}  b {m.intercept_[0]:.3f}  angle of w {ang(m.coef_[0]):.1f} deg  boundary at x2=0.75: x1 = {bx(m.coef_[0], m.intercept_[0]):.3f}")
for nm, f in mk.items():
    r = []
    for X, y in ((Xc, yc), (Xo, yo)):
        m = f().fit(X, y); r.append((ang(m.coef_[0]), bx(m.coef_[0], m.intercept_[0])))
    print(f"{nm:20s} angle {r[0][0]:5.1f} -> {r[1][0]:5.1f} deg (turn {r[1][0] - r[0][0]:5.1f})   boundary at x2=0.75: {r[0][1]:.3f} -> {r[1][1]:.3f}")
print("\n== a far point on the correct side, (-7.5, 3), class 0")
Xp = np.r_[Xc, [out2]]; yp = np.r_[yc, [0]]
for nm, f in mk.items():
    a0 = ang(f().fit(Xc, yc).coef_[0]); a1 = ang(f().fit(Xp, yp).coef_[0]); print(f"{nm:20s} angle {a0:5.1f} -> {a1:5.1f} deg (turn {a1 - a0:5.1f})")
ts = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4]
def turns(pt):
    res = {nm: [] for nm in mk}
    for t in ts:
        for nm, f in mk.items():
            X = np.r_[Xc, [pt * t]] if t else Xc; y = np.r_[yc, [0]] if t else yc
            res[nm].append(ang(f().fit(X, y).coef_[0]))
    return {nm: np.array(v) - v[0] for nm, v in res.items()}
T1, T2 = turns(out), turns(out2)
print("\nturn of the normal vector (degrees) against the distance factor t of the outlier (t = 1 is the problem's point):")
print("t                        ", ts)
for nm in mk: print(f"{nm:25s}", T1[nm].round(1).tolist())
print("same for the point on the correct side:")
for nm in mk: print(f"{nm:25s}", T2[nm].round(1).tolist())

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240, sharey=True)
for ax, TT, ttl in zip(axs, (T1, T2), ('outlier on the wrong side', 'far point on the right side')):
    for (nm, v), c_, ls in zip(TT.items(), (RED, PUR, BLUE), ('-', ':', '--')): ax.plot(ts, v, color=c_, lw=1.5, ls=ls, marker='o', ms=2.8, label=nm)
    ax.set_title(ttl, fontsize=9.2, color=NAVY); ax.set_xlabel('distance factor $t$', fontsize=9.2); ax.tick_params(labelsize=8.8); ax.grid(color=GRID, lw=0.5)
axs[0].set_ylabel('turn of the boundary (degrees)', fontsize=9.2); axs[0].legend(fontsize=8, loc='lower right', handlelength=2.0, borderaxespad=0.2, labelspacing=0.25)
fig.tight_layout(w_pad=0.8)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s18c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s18c1.png') as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))

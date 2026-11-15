# Problem 17C.3: ridge boundary against the number of easy points. Writes fig/s17c3.png, verify/c3.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
np.set_printoptions(precision=3, suppress=True)
x0 = np.array([1, 2, 3, 4, 5, 6.]); y0 = np.array([-1, -1, -1, 1, 1, 1.])
bd = lambda m: -m.intercept_[0] / m.coef_.ravel()[0]
print("== the nine points of Figure 17.6-1: ridge line f = w x + b, margins and losses")
x = np.r_[x0, 12, 14, 16.]; y = np.r_[y0, 1, 1, 1]
r = RidgeClassifier(alpha=1e-6).fit(x[:, None], y); w, b = r.coef_.ravel()[0], r.intercept_[0]
f = w * x + b; z = y * f
print("w = %.4f  b = %.4f  boundary = %.3f" % (w, b, -b / w))
print("x   :", x); print("f   :", f); print("z   :", z); print("(1-z)^2:", (1 - z) ** 2); print("hinge  :", np.maximum(0, 1 - z))
print("sum of (1-z)^2 over the three easy points:", ((1 - z[6:]) ** 2).sum().round(3), " over the six old points:", ((1 - z[:6]) ** 2).sum().round(3))
print("six points alone: ridge line", end=" "); r6 = RidgeClassifier(alpha=1e-6).fit(x0[:, None], y0); print("w = %.4f b = %.4f" % (r6.coef_.ravel()[0], r6.intercept_[0]))
print("squared loss of the six points with the line of the six: ", (((1 - y0 * (r6.coef_.ravel()[0] * x0 + r6.intercept_[0]))) ** 2).sum().round(3), "  with the line of the nine:", ((1 - z[:6]) ** 2).sum().round(3))
print("\n== boundary against the number m of easy points (12, 14, ..., 12 + 2(m-1))")
M = range(0, 9); B = {k: [] for k in ('ridge', 'logistic', 'hinge')}; acc = []
for m in M:
    xx = np.r_[x0, 12 + 2 * np.arange(m)]; yy = np.r_[y0, np.ones(m)]; X1 = xx[:, None]
    rr = RidgeClassifier(alpha=1e-6).fit(X1, yy); lg = LogisticRegression(C=1e4, max_iter=10000).fit(X1, yy); sv = LinearSVC(C=1e4, max_iter=100000).fit(X1, yy)
    B['ridge'].append(bd(rr)); B['logistic'].append(bd(lg)); B['hinge'].append(bd(sv)); acc.append(rr.score(X1, yy))
    print(f"m = {m}  ridge {bd(rr):.3f}  logistic {bd(lg):.3f}  hinge {bd(sv):.3f}   ridge correct {int(round(acc[-1] * len(yy)))} of {len(yy)}")
first = next(m for m in M if acc[m] < 1); print("ridge first misclassifies at m =", first)
fig, ax = plt.subplots(figsize=(5.4, 1.9), dpi=240)
for k, c_, ls in (('ridge', GREEN, '-'), ('logistic', PUR, '--'), ('hinge', BLUE, ':')): ax.plot(list(M), B[k], color=c_, lw=1.6, ls=ls, marker='o', ms=3.5, label=k if k != 'hinge' else 'hinge (LinearSVC)')
ax.axhline(4, color=RED, lw=0.9); ax.text(8.0, 4.34, 'red line = reading 4: a boundary above it loses that reading', color=RED, fontsize=8.3, va='bottom', ha='right')
ax.set_xlabel('number of easy points added (m)', fontsize=9.5); ax.set_ylabel('boundary position', fontsize=9.5); ax.tick_params(labelsize=9); ax.grid(color=GRID, lw=0.5)
ax.set_ylim(2.9, 4.65); ax.legend(fontsize=8.5, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=2.2, borderaxespad=0.2, frameon=False); fig.tight_layout()
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s17c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s17c3.png') as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.9 / w_))

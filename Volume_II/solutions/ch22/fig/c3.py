# Problem 22C.3: decision values against probabilities. Writes fig/s22c3.png, prints to verify/c3.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../../../lib"); os.chdir(HERE + "/..")
import warnings; warnings.simplefilter("error", FutureWarning); warnings.simplefilter("error", DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image

from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
X, y = make_moons(n_samples=400, noise=0.35, random_state=3)
Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
m = SVC(C=1, gamma=1).fit(Xa, ya); f = m.decision_function(Xb)
print("f range on the test samples: %.2f to %.2f;  share with |f| > 1: %.3f;  test accuracy %.3f" % (f.min(), f.max(), (np.abs(f) > 1).mean(), m.score(Xb, yb)))
cal = CalibratedClassifierCV(SVC(C=1, gamma=1), method="sigmoid", cv=5, ensemble=False).fit(Xa, ya)
s = cal.calibrated_classifiers_[0].calibrators[0]; A, B = s.a_, s.b_
P = lambda v: 1 / (1 + np.exp(A * v + B))
print("Platt A = %.2f, B = %.2f;  P(class 1 | f) at f = -2, -1, 0, 1, 2: %s" % (A, B, np.round(P(np.array([-2., -1, 0, 1, 2])), 3)))
print("f at which P = 1/2: %.3f" % (-B / A))
edges = np.array([-3, -1, -0.5, 0, 0.5, 1, 3])
print("bin of f        n   share of class 1   mean calibrated P")
for lo, hi in zip(edges[:-1], edges[1:]):
    k = (f >= lo) & (f < hi)
    if k.sum(): print("[%5.1f,%4.1f)  %4d   %.3f              %.3f" % (lo, hi, k.sum(), yb[k].mean(), P(f[k]).mean()))
print("what 'f > 1' means: share of class 1 among the test samples with f > 1: %.3f (n = %d); with f < -1: %.3f (n = %d)" % (yb[f > 1].mean(), (f > 1).sum(), 1 - yb[f < -1].mean() if False else yb[f < -1].mean(), (f < -1).sum()))
fig, ax = plt.subplots(figsize=(2.9, 1.8), dpi=240)
ee = np.linspace(-2, 2.3, 8); mid = (ee[:-1] + ee[1:]) / 2
sh = [yb[(f >= a) & (f < b)].mean() if ((f >= a) & (f < b)).sum() > 3 else np.nan for a, b in zip(ee[:-1], ee[1:])]
v = np.linspace(-2.2, 2.4, 200)
ax.plot(v, np.clip((v + 1) / 2, 0, 1), color=GREY, ls=":", lw=1.0, label="(f + 1)/2, clipped")
ax.plot(v, P(v), color=GREEN, lw=1.6, label="calibrated sigmoid")
ax.scatter(mid, sh, s=16, color=NAVY, zorder=5, label="share of class 1 in a bin")
ax.axvline(0, color=GRID, lw=0.8); ax.set_xlabel("decision_function f"); ax.set_ylabel("P(class 1)"); ax.set_ylim(-0.03, 1.05)
ax.legend(fontsize=7.8, loc="lower right", handlelength=1.5, borderaxespad=0.2); ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8.6); ax.yaxis.label.set_size(8.6)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig("fig/s22c3.png", dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
with Image.open("fig/s22c3.png") as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 3.0 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 3.0 / w_))

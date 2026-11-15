# Problem 23C.2: the tube swallows the data. Writes fig/s23c2.png, prints to verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../../../lib"); os.chdir(HERE + "/..")
import warnings; warnings.simplefilter("error", FutureWarning); warnings.simplefilter("error", DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.svm import SVR
X = np.array([[1.], [2.], [3.], [4.], [5.]]); y = np.array([3., 4., 6., 7., 10.])
fit = lambda e, C=10: SVR(kernel="linear", C=C, epsilon=e, tol=1e-12).fit(X, y)
print("eps     w       b     SV  support rows   beta          (7 - 2 eps)/4")
for e in (0.5, 1, 1.5, 2, 3, 3.4, 3.49, 3.5, 3.6):
    m = fit(e); print("%-5g %7.4f %7.4f %3d  %-12s %-14s %.4f" % (e, m.coef_[0, 0], m.intercept_[0], len(m.support_), (m.support_ + 1).tolist(), m.dual_coef_[0].round(4).tolist(), max(0, (7 - 2 * e) / 4)))
print("half range of y:", (y.max() - y.min()) / 2, " midrange:", (y.max() + y.min()) / 2)
# the flat line b = 6.5 has residuals -3.5 ... +3.5
print("residuals of the flat line y = 6.5:", (y - 6.5).tolist())
# first-order argument at eps = 3.4: tilt w around the centre x = 3 with b = 6.5
e, C = 3.4, 10.0
for w in (0.0, 0.025, 0.05, 0.1):
    f = 6.5 + w * (X[:, 0] - 3); L = np.maximum(0, np.abs(y - f) - e).sum(); print("eps 3.4, b = 6.5, w = %.3f: sum L = %.3f  objective %.4f" % (w, L, 0.5 * w * w + C * L))
es = np.linspace(0, 4.2, 169); ws = np.array([fit(e).coef_[0, 0] for e in es]); ns = np.array([len(fit(e).support_) for e in es])
fig, ax = plt.subplots(figsize=(3.2, 1.8), dpi=240)
ax.plot(es, ws, color=GREEN, lw=1.8, label="w, fitted")
ee = np.linspace(1.5, 3.5, 20); ax.plot(ee, (7 - 2 * ee) / 4, ls="--", color=RED, lw=1.0, label="(7 − 2ε)/4")
ax.set_xlabel("ε"); ax.set_ylabel("slope w"); ax.set_ylim(0, 1.8); ax.set_xlim(0, 4.2)
a2 = ax.twinx(); a2.step(es, ns, where="mid", color=NAVY, lw=1.0, label="support vectors"); a2.set_ylabel("support vectors", color=NAVY); a2.set_ylim(0, 6); a2.spines["right"].set_visible(True); a2.tick_params(labelsize=8)
ax.axvline(3.5, color=GRID, lw=0.8); ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8.6); ax.yaxis.label.set_size(8.6); a2.yaxis.label.set_size(8.6)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7.8, loc="upper right", handlelength=1.5, borderaxespad=0.2)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig("fig/s23c2.png", dpi=240, bbox_inches="tight", pad_inches=0.04, facecolor="white")
with Image.open("fig/s23c2.png") as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 3.2 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 3.2 / w_))

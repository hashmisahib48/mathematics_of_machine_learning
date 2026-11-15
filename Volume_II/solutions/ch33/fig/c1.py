# Problem 33C.1: the masked class. Writes fig/s33c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

def ls(x, z):
    w = ((x - x.mean()) * (z - z.mean())).sum() / ((x - x.mean()) ** 2).sum(); return w, z.mean() - w * x.mean()
CL = [BLUE, OR, TEAL, PUR, RED]
print("== the six parcels (Example 33.5-1)")
x = np.array([1., 2, 4, 5, 7, 8]); y = np.array([0, 0, 1, 1, 2, 2])
for k in range(3):
    w, b = ls(x, (y == k) * 1.0); print(f"class {k}: slope {w:+.4f}, value at x = 4.5: {w * 4.5 + b:.4f}")
print("1/3 =", round(1 / 3, 4))
for C in (1, 100, 10000):
    m = OneVsRestClassifier(LogisticRegression(C=C, max_iter=5000)).fit(x[:, None], y)
    print(f"logistic, C = {C}: predictions {m.predict(x[:, None])}, decision values at x = 4: {m.decision_function([[4.]])[0].round(2)}",
          " slopes", [round(float(e.coef_[0, 0]), 2) for e in m.estimators_], " intercepts", [round(float(e.intercept_[0]), 2) for e in m.estimators_])

print("\n== five bins in a row, two parcels each (x = 1..10)")
x5 = np.arange(1., 11); y5 = np.repeat(np.arange(5), 2)
L = [ls(x5, (y5 == k) * 1.0) for k in range(5)]
S = np.array([[w * t + b for w, b in L] for t in x5]); pr = S.argmax(1)
print("slopes:", [round(float(w), 4) for w, b in L], " values at the mean 5.5:", [round(float(w * 5.5 + b), 4) for w, b in L])
print("least squares predictions:", pr, " classes ever predicted on x in [0, 11]:", sorted(set(int(v) for v in np.array([[w * t + b for w, b in L] for t in np.linspace(0, 11, 2201)]).argmax(1))))
print("accuracy", (pr == y5).mean())
for C in (1, 100, 10000):
    m = OneVsRestClassifier(LogisticRegression(C=C, max_iter=20000)).fit(x5[:, None], y5)
    p = m.predict(x5[:, None]); print(f"logistic C = {C}: predictions {p}, accuracy {(p == y5).mean():.2f}")

fig, axs = plt.subplots(1, 2, figsize=(4.8, 1.8), dpi=240)
xs = np.linspace(0.5, 10.5, 100)
for k, (w, b) in enumerate(L): axs[0].plot(xs, w * xs + b, color=CL[k], lw=1.2, label=f'bin {k}')
axs[0].set_ylim(-0.4, 0.9); axs[0].set_xlabel('weight $x$'); axs[0].set_ylabel('least-squares score'); axs[0].legend(fontsize=7.6, ncol=2, loc='upper center', handlelength=1.2, columnspacing=0.8)
m = OneVsRestClassifier(LogisticRegression(C=100, max_iter=20000)).fit(x5[:, None], y5)
D = m.decision_function(xs[:, None])
for k in range(5): axs[1].plot(xs, D[:, k], color=CL[k], lw=1.2)
axs[1].set_ylim(-14, 6); axs[1].set_xlabel('weight $x$'); axs[1].set_ylabel('logistic decision value')
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=7.8); a.xaxis.label.set_size(8.2); a.yaxis.label.set_size(8.2)
fig.subplots_adjust(wspace=0.4)
fs = min(t_.get_fontsize() for t_ in fig.findobj(mtext.Text) if t_.get_text().strip() and t_.get_visible())
fig.savefig('fig/s33c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s33c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.8 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.8 / w))

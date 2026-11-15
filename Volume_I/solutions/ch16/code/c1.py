# Problem 16C.1: least squares against logistic regression, with and without a student at 20 hours.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.linear_model import LogisticRegression
sig = lambda z: 1 / (1 + np.exp(-z))
x = np.arange(1., 9); y = np.array([0, 0, 0, 1, 0, 1, 1, 1.])
x2 = np.r_[x, 20.]; y2 = np.r_[y, 1.]
def both(xx, yy):
    ln = np.polyfit(xx, yy, 1)
    m = LogisticRegression(C=np.inf, tol=1e-10, max_iter=5000).fit(xx[:, None], yy)
    return ln, float(m.intercept_[0]), float(m.coef_[0, 0])
for nm, (xx, yy) in (("8 students", (x, y)), ("with the student at 20 h", (x2, y2))):
    ln, b, w = both(xx, yy)
    print(nm + ":")
    print("   line  y = %.4f x + %.4f   crosses 0.5 at x = %.3f   at 1 h: %.3f   at 8 h: %.3f   at 20 h: %.3f"
          % (ln[0], ln[1], (0.5 - ln[1]) / ln[0], np.polyval(ln, 1), np.polyval(ln, 8), np.polyval(ln, 20)))
    print("   logistic  b = %.4f  w = %.4f  boundary %.3f  p(1 h) = %.4f  p(8 h) = %.4f  p(4 h) = %.4f  p(5 h) = %.4f"
          % (b, w, -b / w, sig(b + w), sig(b + 8 * w), sig(b + 4 * w), sig(b + 5 * w)))
    r = yy - np.polyval(ln, xx); print("   squared residual of the extra student for the line:", (r[-1] ** 2) if len(xx) == 9 else "-")
ln1, b1, w1 = both(x, y); ln2, b2, w2 = both(x2, y2)
print("students misclassified by the line (threshold 0.5), before / after:", ((np.polyval(ln1, x) > .5) != (y == 1)).sum(), ((np.polyval(ln2, x2) > .5) != (y2 == 1)).sum())
print("students misclassified by the logistic model, before / after:", ((sig(b1 + w1 * x) > .5) != (y == 1)).sum(), ((sig(b2 + w2 * x2) > .5) != (y2 == 1)).sum())
J2 = lambda b, w: float(np.sum(np.logaddexp(0, b + w * x2) - y2 * (b + w * x2)))
print("loss of the new student at the old logistic fit: ln(1 + e^-z) =", np.log1p(np.exp(-(b1 + 20 * w1))), " (the others sum to", float(np.sum(np.logaddexp(0, b1 + w1 * x) - y * (b1 + w1 * x))), ")")

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240, sharey=True)
xx = np.linspace(-1, 21, 200)
for ax, (xs_, ys_), (ln, b, w), ttl in ((axs[0], (x, y), (ln1, b1, w1), '8 students'), (axs[1], (x2, y2), (ln2, b2, w2), 'plus one at 20 hours')):
    ax.plot(xx, np.polyval(ln, xx), color=GREY, lw=1.2, ls='--', label='least squares'); ax.plot(xx, sig(b + w * xx), color=GREEN, lw=1.6, label='logistic')
    ax.scatter(xs_, ys_, s=14, color=BLUE, edgecolor='white', zorder=5); ax.axhline(0.5, color=GRID, lw=0.6)
    ax.set_xlabel('hours $x$  (' + ttl + ')'); ax.set_ylim(-0.4, 1.5); ax.set_yticks([0, 0.5, 1]); ax.set_xlim(-1, 21)
axs[0].set_ylabel('$y$'); axs[0].legend(fontsize=7.6, loc='lower right', handlelength=1.8)
fig.tight_layout(); fs = min(t_.get_fontsize() for t_ in fig.findobj(matplotlib.text.Text) if t_.get_text().strip()); save(fig, 'fig/s16c1.png'); print("smallest text", fs)

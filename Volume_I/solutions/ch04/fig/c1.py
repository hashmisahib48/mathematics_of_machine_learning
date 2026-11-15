# Problem 4C.1: zero correlation, strong dependence. Writes fig/s4c1.png, verify/c1.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *

rng = np.random.default_rng(3); x = rng.uniform(-2, 2, 120); y = x ** 2 + rng.normal(0, 0.2, 120)   # the book's data
xm, ym = x.mean(), y.mean(); dx, dy = x - xm, y - ym
print("mean x = %.4f   mean y = %.4f" % (xm, ym))
print("covariance (x, y) = %.4f   Var x = %.4f   Var y = %.4f" % ((dx * dy).sum() / 119, dx.var(ddof=1), dy.var(ddof=1)))
r = lambda a, b: np.corrcoef(a, b)[0, 1]
print("r(x, y) = %.3f   r(x^2, y) = %.3f   r(|x|, y) = %.3f" % (r(x, y), r(x ** 2, y), r(np.abs(x), y)))
pos, neg = x > 0, x < 0
print("sum of dx*dy over x > 0: %.2f   over x < 0: %.2f   total %.2f" % ((dx * dy)[pos].sum(), (dx * dy)[neg].sum(), (dx * dy).sum()))
print("quadrant counts of (dx, dy): ++ %d  -+ %d  +- %d  -- %d" % ((pos & (dy > 0)).sum(), (neg & (dy > 0)).sum(), (pos & (dy < 0)).sum(), (neg & (dy < 0)).sum()))
print("exact: Cov(x, x^2) = E[x^3] - E[x] E[x^2] = 0 - 0 for x uniform on [-2, 2]; E[x^2] = %.4f (4/3)" % (4 / 3))
print("R^2 of y on x: %.3f   on x^2: %.3f" % (r(x, y) ** 2, r(x ** 2, y) ** 2))

fig, axs = plt.subplots(1, 2, figsize=(4.6, 1.8), dpi=240)
for ax, u, lab in ((axs[0], x, '$x$'), (axs[1], x ** 2, '$x^2$')):
    ax.scatter(u, y, s=9, color=BLUE, edgecolor='white', lw=0.3); ax.set_xlabel(lab, fontsize=8); ax.tick_params(labelsize=8)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.text(0.5, 0.95, '$r$ = %s' % ('%.3f' % r(u, y)).replace('-', '−'), transform=ax.transAxes, color=RED, fontsize=8, va='top', ha='center')
axs[0].set_ylabel('$y$', fontsize=8)
fig.tight_layout(w_pad=1.2); save(fig, 'fig/s4c1.png')

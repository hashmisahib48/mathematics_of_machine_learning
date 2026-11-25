"""Figure 3.1-1: least squares on five test scores (sidebar size): the line, and its errors on their own scale."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
x = np.array([1, 2, 3, 4, 5.]); y = np.array([52, 57, 61, 68, 72.])
w, b = 5.1, 46.7; e = y - (w * x + b)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(2.05, 2.55), dpi=300, gridspec_kw={'height_ratios': [1.6, 1]}, sharex=True)
xx = np.linspace(0.6, 5.4, 2); ax.plot(xx, w * xx + b, color=GREEN, lw=1.3)
for xi, yi in zip(x, y): ax.plot([xi, xi], [yi, w * xi + b], color=RED, lw=2.2, solid_capstyle='butt')
ax.plot(x, y, 'o', ms=3.6, color=BLUE, zorder=3)
ax.set_ylabel('test score, $y$', fontsize=7.5); ax.tick_params(labelsize=7, length=2.5); ax.set_yticks([50, 60, 70])
ax.text(0.7, 73, r'$\hat y = 5.1x + 46.7$', fontsize=7.5, color=GREEN); ax.set_ylim(48, 76)
ax2.axhline(0, color=GREEN, lw=1.0)
ax2.bar(x, e, width=0.22, color=RED)
for xi, ei in zip(x, e): ax2.text(xi, ei + (0.12 if ei >= 0 else -0.12), f'{ei:+.1f}'.replace('-', '−'), ha='center', va='bottom' if ei >= 0 else 'top', fontsize=6.3, color=RED)
ax2.set_ylim(-1.6, 1.6); ax2.set_yticks([-1, 0, 1]); ax2.set_ylabel('error, $e_i$', fontsize=7.5)
ax2.set_xlabel('hours studied, $x$', fontsize=7.5); ax2.tick_params(labelsize=7, length=2.5); ax2.set_xticks([1, 2, 3, 4, 5]); ax2.set_xlim(0.5, 5.5)
fig.tight_layout(pad=0.2, h_pad=0.3); fig.savefig(os.path.join(HERE, 'png', 'f31.png'), dpi=300, facecolor='white'); plt.close(fig)

"""Figure 3.2-1: least squares as a right angle (projection of y~ on the plane of the feature columns)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.patches import Polygon
fig, ax = new(2.3, 1.6)
P = lambda x, y, z: (x + 0.55 * y, 0.35 * y + z)          # oblique view of 3-D points
plane = [P(-0.3, -0.4, 0), P(3.6, -0.4, 0), P(3.6, 2.6, 0), P(-0.3, 2.6, 0)]
ax.add_patch(Polygon(plane, closed=True, fc=LB, ec=BLUE, lw=0.8))
O = P(0, 0, 0); yh = P(2.4, 1.4, 0); yt = P(2.4, 1.4, 1.9)
arr(ax, O, P(1.8, 0, 0), c=NAVY, lw=1.1, ms=8); ax.text(*P(1.85, -0.05, 0), r'$\tilde{\mathbf{x}}_1$', fontsize=10.5, va='top', color=NAVY)
arr(ax, O, P(0, 1.6, 0), c=NAVY, lw=1.1, ms=8); ax.text(*P(-0.05, 1.7, 0), r'$\tilde{\mathbf{x}}_2$', fontsize=10.5, ha='right', va='bottom', color=NAVY)
arr(ax, O, yt, c=GREEN, lw=1.3, ms=9); ax.text(yt[0] - 0.08, yt[1] + 0.05, r'$\tilde{\mathbf{y}}$', fontsize=11, ha='right', color=GREEN)
arr(ax, O, yh, c=OR, lw=1.3, ms=9); ax.text(yh[0] - 0.05, yh[1] - 0.1, r'$\tilde X\mathbf{w}$', fontsize=10.5, ha='right', va='top', color='#B9770E')
ax.plot([yh[0], yt[0]], [yh[1], yt[1]], color=RED, lw=1.3, ls=(0, (3, 2)))
ax.text(yt[0] - 0.12, (yh[1] + yt[1]) / 2, r'$\mathbf{e}$', fontsize=11, color=RED, ha='right')
s = 0.18; a = P(2.4 - s, 1.4, 0); b = P(2.4 - s, 1.4, s); c = P(2.4, 1.4, s)
ax.plot([a[0], b[0], c[0]], [a[1], b[1], c[1]], color=RED, lw=0.8)
ax.text(3.45, 0.98, 'plane of the\nfeature columns', fontsize=8, ha='left', va='bottom', color=BLUE, linespacing=1.05)
ax.set_xlim(-0.8, 5.7); ax.set_ylim(-0.4, 2.7)
save(fig, os.path.join(HERE, 'png', 'f32.png'))

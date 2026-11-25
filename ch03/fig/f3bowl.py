"""Figure 3.2-1: J(w, b) for the five students over a rectangle of the (w, b) plane, with contours on the floor."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
x = np.array([1, 2, 3, 4, 5.]); y = np.array([52, 57, 61, 68, 72.]); n = len(x)
w0, b0 = 5.1, 46.7; Jmin = ((y - w0 * x - b0) ** 2).sum()
W, B = np.meshgrid(np.linspace(3.4, 6.8, 60), np.linspace(36, 58, 60))
J = ((y[None, None, :] - W[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)
fig = plt.figure(figsize=(2.05, 1.95), dpi=300); ax = fig.add_axes([0.0, 0.02, 0.95, 0.98], projection='3d', computed_zorder=False)
ax.plot_surface(W, B, J, cmap='viridis', rstride=2, cstride=2, linewidth=0.1, edgecolor='#ffffff50', alpha=0.95, zorder=2)
ax.contour(W, B, J, zdir='z', offset=0, levels=[10, 30, 100, 300, 600], colors='#777777', linewidths=0.5, zorder=1)
ax.scatter([w0], [b0], [Jmin], color=RED, s=14, depthshade=False, zorder=10)
ax.set_xlabel('$w$', fontsize=9, labelpad=-9); ax.set_ylabel('$b$', fontsize=9, labelpad=-9)
ax.tick_params(labelsize=7, pad=-4); ax.set_xticks([4, 5, 6]); ax.set_yticks([40, 48, 56]); ax.set_zticks([0, 500, 1000])
ax.view_init(26, -52); ax.set_zlim(0, 1400); ax.set_box_aspect((1, 1, 0.8))
fig.text(0.90, 0.79, '$J$', fontsize=9, ha='center', va='center')
fig.savefig(os.path.join(HERE, 'png', 'f3bowl.png'), dpi=300, facecolor='white'); plt.close(fig)
print(Jmin, J.min(), J.max())

"""Figure 3.2-1: J(w, b) for the five students, drawn over the region J <= Jmax so that the rim is a level curve (an ellipse)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
x = np.array([1, 2, 3, 4, 5.]); y = np.array([52, 57, 61, 68, 72.]); n = len(x)
H = np.array([[(x * x).sum(), x.sum()], [x.sum(), n]])          # J = Jmin + d^T H d,  d = (w - 5.1, b - 46.7)
lam, V = np.linalg.eigh(H); Jmin, w0, b0, Jtop = 1.9, 5.1, 46.7, 60.0
r = np.linspace(0, 1, 40); t = np.linspace(0, 2 * np.pi, 90); R, T = np.meshgrid(r, t)
s = np.sqrt(Jtop - Jmin) * R; u = s * np.cos(T) / np.sqrt(lam[0]); v = s * np.sin(T) / np.sqrt(lam[1])
W = w0 + V[0, 0] * u + V[0, 1] * v; B = b0 + V[1, 0] * u + V[1, 1] * v
J = ((y[None, None, :] - W[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)
fig = plt.figure(figsize=(2.05, 1.95), dpi=300); ax = fig.add_axes([-0.04, 0.03, 0.98, 1.0], projection='3d', computed_zorder=False)
ax.plot_surface(W, B, J, cmap='viridis', rstride=2, cstride=2, linewidth=0.1, edgecolor='#ffffff40', alpha=0.95, zorder=2)
for lev in [10, 30, 60]:
    ss = np.sqrt(lev - Jmin); uu = ss * np.cos(t) / np.sqrt(lam[0]); vv = ss * np.sin(t) / np.sqrt(lam[1])
    ax.plot(w0 + V[0, 0] * uu + V[0, 1] * vv, b0 + V[1, 0] * uu + V[1, 1] * vv, 0 * t, color='#888888', lw=0.5, zorder=1)
ax.scatter([w0], [b0], [Jmin], color=RED, s=12, depthshade=False, zorder=10)
ax.plot([w0, w0], [b0, b0], [0, Jmin], color=RED, lw=0.6, zorder=10)
ax.set_xlabel('$w$', fontsize=7.5, labelpad=-8); ax.set_ylabel('$b$', fontsize=7.5, labelpad=-8); ax.set_zlabel('$J$', fontsize=7.5, labelpad=-9)
ax.tick_params(labelsize=5.6, pad=-3); ax.set_xticks([3, 5, 7]); ax.set_yticks([40, 45, 50, 55]); ax.set_zticks([0, 20, 40, 60]); ax.view_init(28, -35); ax.set_zlim(0, Jtop)
ax.set_box_aspect((1, 1, 0.8))
fig.savefig(os.path.join(HERE, 'png', 'f3bowl.png'), dpi=300, facecolor='white'); plt.close(fig)
print(lam, V[:, 0], W.min(), W.max(), B.min(), B.max())

# Shared helpers of the Chapter 36 figure scripts.
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib')
os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning)
warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from PIL import Image

plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9,
                     'legend.fontsize': 8.5})
LO = '#FDEBD0'; LT = '#DDF1F2'          # light orange, light teal (cell shading)
SHADE = [LB, LO, LT, '#F1EAF7', '#E3F4EA', '#FBE6E3']
PTCOL = [BLUE, OR, TEAL, MAG, PUR, GREY]
MARK = ['o', 's', '^', 'D', 'v', 'P']
CHK = {}
NUM = {}

# the eight customers of the chapter (x east, y north, in km)
X8 = np.array([[3, 7], [1, 3], [1, 4], [3, 8], [4, 0], [2, 1], [5, 8], [1, 8]], float)


def sv(fig, name, printed_px):
    """save and record the smallest printed font size (pt)"""
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    with Image.open(name) as im:
        w = im.size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}


def lloyd_path(X, mu0, maxit=20):
    """list of (centres, labels, J_assign) per assignment step, Lloyd's algorithm"""
    mu = np.array(mu0, float); out = []; prev = None
    for _ in range(maxit):
        d2 = ((X[:, None] - mu[None]) ** 2).sum(2); lab = d2.argmin(1)
        out.append((mu.copy(), lab.copy(), float(d2[np.arange(len(X)), lab].sum())))
        if prev is not None and (lab == prev).all():
            break
        prev = lab; mu = np.array([X[lab == k].mean(0) for k in range(len(mu))])
    return out


def inertia_of(X, lab, mu):
    return float(((X - mu[lab]) ** 2).sum())


def voronoi_shade(ax, mu, xlim, ylim, cols=None, n=300):
    """shade the Voronoi cells of the centres mu"""
    gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    g = np.c_[gx.ravel(), gy.ravel()]
    z = ((g[:, None] - mu[None]) ** 2).sum(2).argmin(1).reshape(gx.shape)
    cm = ListedColormap((cols or SHADE)[:len(mu)])
    ax.pcolormesh(gx, gy, z, cmap=cm, vmin=0, vmax=len(mu) - 1, shading='auto', zorder=0)
    ax.contour(gx, gy, z, levels=np.arange(len(mu)) + 0.5, colors=[GREEN], linewidths=1.1, zorder=1)


def pts(ax, X, lab, s=26, truth=None, **k):
    for c in sorted(set(lab)):
        m = lab == c
        mk = MARK[c % len(MARK)] if truth is None else None
        if truth is None:
            ax.scatter(X[m, 0], X[m, 1], s=s, color=PTCOL[c % 6], marker=mk, edgecolor='white', lw=0.6, zorder=5, **k)
        else:
            for t in sorted(set(truth[m])):
                mm = m & (truth == t)
                ax.scatter(X[mm, 0], X[mm, 1], s=s, color=PTCOL[c % 6], marker=MARK[t % len(MARK)],
                           edgecolor='white', lw=0.6, zorder=5, **k)


def centres(ax, mu, s=80):
    ax.scatter(mu[:, 0], mu[:, 1], s=s, marker='X', color=GREEN, edgecolor='white', lw=0.9, zorder=7)


def grid(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)

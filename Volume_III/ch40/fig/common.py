# shared set-up for the figure scripts of Chapter 40
import sys, os, json, warnings, matplotlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code')
os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GM
from sklearn.exceptions import ConvergenceWarning
from people import people_2d, heights_1d, GROUPS
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5})
TEAL3 = TEAL
GCOL = [BLUE, OR, TEAL]            # colours of the three groups / components
GMK = ['o', 's', '^']
CHK = {}


def sv(fig, name, printed_px):
    """save and record the smallest printed font size (pt)"""
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    with Image.open(name) as im:
        w = im.size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}


def grid(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def ell(ax, mean, cov, nstd=2.0, color=GREEN, lw=1.5, ls='-', **k):
    """ellipse of the Gaussian N(mean, cov): contour at nstd standard deviations (Mahalanobis distance nstd)"""
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    ax.add_patch(Ellipse(mean, 2 * nstd * np.sqrt(w[-1]), 2 * nstd * np.sqrt(w[0]), angle=ang, fc='none', ec=color, lw=lw, ls=ls, **k))


def blend(R, cols=GCOL):
    """colour of each point = responsibility-weighted mix of the component colours"""
    from matplotlib.colors import to_rgb
    C = np.array([to_rgb(c) for c in cols])
    return np.clip(R @ C, 0, 1)


def dump(name, d):
    p = 'verify/' + name
    json.dump(d, open(p, 'w'), indent=1, default=float)

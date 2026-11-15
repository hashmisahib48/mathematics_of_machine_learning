# Shared set-up of the Chapter 41 figure scripts.
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib')
os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning)
warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, load_wine
from sklearn.metrics import (adjusted_rand_score as ari, adjusted_mutual_info_score as ami,
                             silhouette_score, silhouette_samples, calinski_harabasz_score as chs,
                             davies_bouldin_score as dbs, rand_score, normalized_mutual_info_score as nmi)

plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9,
                     'legend.fontsize': 8.5})
LO = '#FDEBD0'; LT = '#DDF1F2'
COL = [BLUE, OR, TEAL, MAG, PUR, GREY]
MARK = ['o', 's', '^', 'D', 'v', 'P']
CHK = {}
NUM = {}

# the eight customers of the chapter, their shop segments and the k-means clusters
X8 = np.array([[1, 1], [2, 3], [3, 1], [4, 3], [7, 2], [8, 4], [3, 7], [5, 8]], float)
SEG = np.array([0, 0, 0, 1, 1, 1, 2, 2])
CLU = np.array([0, 0, 0, 0, 1, 1, 2, 2])
NAMES = list('ABCDEFGH')

# the four blobs of Chapter 36 (Figure 36.11-1)
CENT = [(-5, -3), (-3, 4), (4, -4), (5, 3)]
XB, YB = make_blobs(400, centers=CENT, cluster_std=1.5, random_state=3)
XM, YM = make_moons(400, noise=0.07, random_state=0)


def sv(fig, name, printed_px):
    """save and record the smallest printed font size (pt)"""
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    with Image.open(name) as im:
        w = im.size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}


def pts(ax, X, lab, s=8, **k):
    for j in np.unique(lab):
        m = lab == j
        ax.scatter(X[m, 0], X[m, 1], s=s, color=COL[j % 6], marker=MARK[j % 6], lw=0, **k)


def grid(ax):
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def dump(name, d):
    json.dump(d, open('verify/' + name, 'w'), indent=1, default=float)

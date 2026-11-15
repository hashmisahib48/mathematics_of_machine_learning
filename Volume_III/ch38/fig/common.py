# shared helpers for figs.py and probs.py (Chapter 38)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code')
os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning)
warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.lines import Line2D
CL = [BLUE, OR, TEAL, PUR, MAG, '#8C6D1F', '#4C8C2B', '#5B6C8F']   # cluster colours
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']                       # and markers
NOISE = '#7A7A7A'
LO = '#FDEBD0'


def scatter_labels(ax, X, lab, s=14, alpha=1.0, noise_s=14, lw=0.4):
    """clusters in colour with their own marker, noise as grey crosses"""
    for k in sorted(set(lab)):
        m = lab == k
        if k == -1:
            ax.scatter(X[m, 0], X[m, 1], s=noise_s, marker='x', color=NOISE, lw=0.8, zorder=3)
        else:
            ax.scatter(X[m, 0], X[m, 1], s=s, marker=MK[k % 8], color=CL[k % 8],
                       edgecolor='white', lw=lw, alpha=alpha, zorder=4)


def grid(ax):
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def clean(ax, labels=False):
    ax.set_xticks([])
    ax.set_yticks([])
    if not labels:
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_color('#999999')


class Saver:
    """save a figure and record the smallest printed font size (pt)"""
    def __init__(self):
        self.chk = {}

    def __call__(self, fig, name, printed_px):
        fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text)
                 if t.get_text().strip() and t.get_visible())
        save(fig, name)
        from PIL import Image
        with Image.open(name) as im:
            w = im.size[0] / 240
        self.chk[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2),
                          'min_font_design': round(fs, 1),
                          'min_font_printed': round(fs * printed_px / 96 / w, 1)}

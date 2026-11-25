"""Listing headers: teal ribbon with the listing number in a gold-rimmed navy medallion (one image per listing)."""
import sys, os, re, glob
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib import font_manager as fm
for f in glob.glob('/root/.fonts/JetBrainsMono*.ttf'): fm.fontManager.addfont(f)
HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 3.3, 0.34
def head(no, file, out):
    fig = plt.figure(figsize=(W, H), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    ax.add_patch(Polygon([(0.05, 0.04), (W, 0.04), (W - 0.15, 0.19), (W, H), (0.05, H)], closed=True, fc='#1A8A8A', ec='none'))
    ax.add_patch(Circle((0.17, 0.19), 0.16, fc='#0B1F3A', ec='#E2C26A', lw=1.3, zorder=3))
    ax.text(0.17, 0.195, no, fontsize=8, family='Liberation Serif', weight='bold', color='white', ha='center', va='center', zorder=4)
    ax.text(0.42, 0.245, 'LISTING', fontsize=7.2, family='Carlito', weight='bold', color='white', va='baseline')
    ax.text(1.0, 0.245, file, fontsize=7.6, family='JetBrains Mono', color='#FFF4C2', va='baseline')
    fig.savefig(out, dpi=300, transparent=True); plt.close(fig)
src = open(os.path.join(HERE, '..', 'build', 'ch03.js')).read()
for no, file in re.findall(r"add\((?:listing|codeOnly)\('([\d.]+)', '([^']+)'", src):
    head(no, file, os.path.join(HERE, '..', 'assets', f'lsthead_{no}.png'))
print('ok')

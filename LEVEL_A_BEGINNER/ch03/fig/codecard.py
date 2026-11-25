"""One Dark code-card pieces: title bar per listing (traffic lights, centred file name), bottom cap, output caps."""
import os, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
HERE = os.path.dirname(os.path.abspath(__file__)); AS = os.path.join(HERE, '..', 'assets')
W = 4.4; R = 0.07
def cap(out, h, fill, top=True, draw=None):
    fig = plt.figure(figsize=(W, h), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(h, 0); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0, 0 if top else -R), W, h + R, boxstyle=f'round,pad=0,rounding_size={R}', fc=fill, ec='none', mutation_aspect=1))
    if top: ax.add_patch(Rectangle((0, h - 0.01), W, 0.02, fc=fill, ec='none'))
    if draw: draw(ax, h)
    fig.savefig(out, dpi=300, transparent=True); plt.close(fig)
def bar(no, file):
    def d(ax, h):
        for k, c in enumerate(['#ff5f56', '#ffbd2e', '#27c93f']): ax.add_patch(Circle((0.13 + k * 0.12, h / 2), 0.035, fc=c, ec='none'))
        ax.text(W / 2, h / 2, file, fontsize=6.6, family='DejaVu Sans Mono', color='#9da5b4', ha='center', va='center')
        ax.text(W - 0.12, h / 2, f'Listing {no}', fontsize=6.2, family='DejaVu Sans', weight='bold', color='#7f848e', ha='right', va='center')
    cap(os.path.join(AS, f'cbar_{no}.png'), 0.22, '#21252b', True, d)
src = open(os.path.join(HERE, '..', 'build', 'chapter.js')).read()
for no, file in re.findall(r"add\((?:listing|codeOnly)\('([\d.]+)', '([^']+)'", src): bar(no, file)
cap(os.path.join(AS, 'cbot.png'), 0.07, '#282c34', False)
cap(os.path.join(AS, 'otop.png'), 0.2, '#21252b', True,
    lambda ax, h: ax.text(0.14, h / 2 + 0.02, 'OUTPUT', fontsize=5.8, family='DejaVu Sans', weight='bold', color='#7f848e', va='center'))
cap(os.path.join(AS, 'obot.png'), 0.07, '#21252b', False)
print('ok')

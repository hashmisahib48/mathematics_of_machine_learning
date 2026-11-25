"""Sidebar figures: the picture with its caption under it, as one image 1.75 in wide."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams['mathtext.fontset'] = 'stix'
W = 1.75; DPI = 300
def cap(src, label, text, out):
    im = Image.open(os.path.join(HERE, 'png', src)).convert('RGB'); w, h = im.size; ih = W * h / w
    probe = plt.figure(figsize=(W, 1), dpi=DPI); r = probe.canvas.get_renderer()
    words = text.split(); lines, cur = [], ''
    first = label + '  '
    def width(s): t = probe.text(0, 0, s, fontsize=8, family='Liberation Serif'); x = t.get_window_extent(renderer=r).width / DPI; t.remove(); return x
    lw = width(first)
    for wd in words:
        trial = (cur + ' ' + wd).strip()
        if width(trial) + (lw if not lines else 0) <= W - 0.02 or not cur: cur = trial
        else: lines.append(cur); cur = wd
    lines.append(cur); plt.close(probe)
    lh = 8 / 72 * 1.2; ch = 0.06 + lh * len(lines)
    fig = plt.figure(figsize=(W, ih + ch), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(ih + ch, 0); ax.axis('off')
    ax.imshow(im, extent=(0, W, ih, 0), aspect='auto')
    y = ih + 0.06 + lh * 0.8
    ax.text(0, y, label, fontsize=8, family='Liberation Serif', weight='bold', color='#1F5FBF', va='baseline')
    for k, ln in enumerate(lines):
        ax.text(lw if k == 0 else 0, y + k * lh, ln, fontsize=8, family='Liberation Serif', color='#222222', va='baseline')
    fig.savefig(os.path.join(HERE, 'png', out), dpi=DPI, facecolor='white'); plt.close(fig)
cap('f61.png', 'Figure 6.4-1', 'The six training points of Example 6.4-1 and the new point $q$. The circles enclose its nearest neighbor and its three nearest neighbors.', 'f61s.png')
cap('f62.png', 'Figure 6.2-1', 'Decision regions of $k$-NN on two iris features. With $k=1$ the boundary follows every point; with $k=25$ it is smoother.', 'f62s.png')
cap('f63.png', 'Figure 6.7-1', 'Training and test $R^2$ of $k$-NN regression on the diabetes data. Small $k$ overfits; the test score is best for $k$ between 20 and 40.', 'f63s.png')
cap('pj62.png', 'Figure P6-2', 'The first ten of the 1797 handwritten digits, each an 8 × 8 image of 64 gray levels.', 'pj62s.png')

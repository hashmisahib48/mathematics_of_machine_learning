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
cap('f41.png', 'Figure 4.1-1', 'Five points that follow a curve. The least-squares line misses the pattern; a quadratic in $x$, fitted as a linear model in $x$ and $x^2$, follows it.', 'f41s.png')
cap('f42.png', 'Figure 4.2-1', 'A polynomial of degree 9 fitted to 12 points. Least squares oscillates; a small ridge penalty keeps the weights small and the curve close to the true one.', 'f42s.png')
cap('f43.png', 'Figure 4.7-1', 'Ridge path for the diabetes data: the ten weights, on standardized features, as the penalty grows. All tend to zero, but at different rates; age, sex, s1, s2 and s3 change sign.', 'f43s.png')
cap('pj43.png', 'Figure P4-3', 'The Engel data: food expenditure against income, with the least-squares line, and the number of households in each income band.', 'pj43s.png')
cap('pj44.png', 'Figure P4-4', 'The diabetes data: disease progression against body mass index, with the least-squares line.', 'pj44s.png')
cap('pj41.png', 'Figure P4-1', 'Ridge path for the Longley data. The weights of the nearly dependent features change sign at small penalties and settle as the penalty grows.', 'pj41s.png')

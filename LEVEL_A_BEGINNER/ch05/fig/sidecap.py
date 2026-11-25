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
cap('f51.png', 'Figure 5.2-1', 'Error of the slope for the five students at four learning rates, 0.1, 0.25, 0.45 and 0.55. The factor $1-4\\eta$ sets the behavior: steady decay, one step, oscillating decay, and divergence.', 'f51s.png')
cap('f52.png', 'Figure 5.2-2', 'Twenty-five steps on the deliveries, on the same scale in both panels. The eigenvalue ratio of $A$ is 17.4 for raw and 4.0 for standardized features: the steps zigzag in the narrow raw valley.', 'f52s.png')
cap('f53.png', 'Figure 5.2-3', 'Stochastic and batch descent for the five students, 12 epochs, rate 0.05. An epoch of five stochastic steps moves further than one batch step (no factor $1/n$ in Eq. 5.2-9).', 'f53s.png')
cap('pj5_1.png', 'Figure P5-1', 'Stack loss against air flow with the least-squares line (top), and the range of the raw values of the three operating conditions (bottom).', 'pj5_1s.png')
cap('pj5_2.png', 'Figure P5-2', 'Diabetes data. Top: the measure of disease progression of the 442 patients. Bottom: progression against the body mass index.', 'pj5_2s.png')
cap('pj5_3.png', 'Figure P5-3', 'Longley data. Top: GNP, population and price deflator, standardized, rise together from 1947 to 1962. Bottom: the number employed.', 'pj5_3s.png')
cap('pj5_4.png', 'Figure P5-4', 'Engel data: the logarithm of food expenditure against the logarithm of income for 235 households, with the least-squares line.', 'pj5_4s.png')
cap('pj5_5.png', 'Figure P5-5', 'One epoch of twelve observations, shuffled and cut into groups of size $m$; each group makes one update. The case $m=1$ is stochastic descent and $m=12$ is batch descent.', 'pj5_5s.png')

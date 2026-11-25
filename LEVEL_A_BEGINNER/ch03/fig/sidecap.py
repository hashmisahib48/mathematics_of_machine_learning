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
cap('f31.png', 'Figure 3.1-1', 'Least squares on five test scores. Top: the data and the fitted line. Bottom: the error $e_i$ of each student, drawn to a larger scale; the line makes the sum of their squares as small as possible.', 'f31s.png')
cap('f3bowl.png', 'Figure 3.2-1', 'The sum of squared errors $J(w,b)$ of the five students, The surface is a long trough because a larger slope can be offset by a smaller intercept; the floor contours are ellipses along $b=62-3w$. The lowest point (red), $J=1.9$, lies at $w=5.1$, $b=46.7$.', 'f3bowls.png')
cap('f32.png', 'Figure 3.2-2', 'Least squares as a right angle. The fitted values $\\tilde X\\mathbf{w}$ are the point of the plane of the feature columns nearest to $\\tilde{\\mathbf{y}}$; the error vector $\\mathbf{e}$ meets the plane at a right angle.', 'f32s.png')
cap('pj1.png', 'Figure P3-1', 'Stack loss of the ammonia plant. Top: against air flow, with the one-feature line. Bottom: errors of the three-feature fit; days 21, 4 and 3 are numbered.', 'pj1s.png')
cap('pj2.png', 'Figure P3-2', 'Longley data, 1947 to 1962. Top: employment and the six-feature fit. Bottom: population against GNP, two features that are almost a straight line.', 'pj2s.png')
cap('pj3.png', 'Figure P3-3', 'Engel data, 235 households. Top: food expenditure against income; the scatter widens as income grows. Bottom: the same data on logarithmic scales.', 'pj3s.png')
cap('pj4.png', 'Figure P3-4', 'Linnerud data, 20 men. Top: sit-ups against waist, with the one-feature line. Bottom: waist against weight, two strongly related features.', 'pj4s.png')

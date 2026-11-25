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
cap('f21.png', 'Figure 2.2-1', 'Deviations from the means for the five students. Each rectangle has area $(x_i-\\bar{x})(y_i-\\bar{y})$; none is negative, and their sum is $S_{xy}=51$.', 'f21s.png')
cap('f22.png', 'Figure 2.3-1', 'Two vectors in the plane. Their dot product is 24, their lengths are 5, and the angle between them is 16.3°.', 'f22s.png')
cap('f23.png', 'Figure 2.4-1', 'The function $J(w)$ of Listing 2.3. The tangent at $w=1$ has slope $-34$; at the minimum, $w=2.2143$, the slope is zero.', 'f23s.png')
cap('f24.png', 'Figure 2.4-2', 'Contours of $J=w_1^2+3w_2^2$. The gradient (red) points uphill, across the contours; its negative (green) points toward the minimum.', 'f24s.png')
cap('f25.png', 'Figure 2.5-1', 'Two normal densities. The mean $\\mu$ sets the center and the standard deviation $\\sigma$ the spread; each curve encloses an area of 1.', 'f25s.png')
cap('f26.png', 'Figure 2.6-1', 'Two distances between (1, 1) and (4, 4): the straight line, and the path along the grid.', 'f26s.png')

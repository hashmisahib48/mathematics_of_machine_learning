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
cap('f11.png', 'Figure 1.1-1', 'The two kinds of supervised learning. Top: regression predicts a number. Bottom: classification predicts a class; the three species of iris form groups in petal length and width.', 'f11s.png')
cap('f13.png', 'Figure 1.2-1', 'A data set in the form scikit-learn expects: a table $X$ of features with one row per observation, and a column $y$ of targets.', 'f13s.png')
cap('f14.png', 'Figure 1.3-1', 'Polynomials of degree 1, 3 and 9 fitted to twelve noisy points. Degree 1 underfits, degree 9 passes close to every point but oscillates between them.', 'f14s.png')
cap('f15.png', 'Figure 1.4-1', 'The workflow of this book: split the data, fit on the training set, predict and score on the test set.', 'f15s.png')
cap('pj1.png', 'Figure P1-1', 'Wine data, 178 wines. Top: proline against color intensity. Bottom: the same points after each feature is rescaled to run from 0 to 1.', 'pj1s.png')
cap('pj2.png', 'Figure P1-2', 'Breast cancer data, 569 tumors. Top: mean texture against mean radius. Bottom: histogram of the mean concave points.', 'pj2s.png')
cap('pj3.png', 'Figure P1-3', 'Diabetes data, 442 patients. Top: progression against s5, with the prediction of a tree of depth 1 (green). Bottom: histogram of the progression.', 'pj3s.png')
cap('pt1.png', 'Table P1-1', 'Six wines and a new wine of unknown class.', 'pt1s.png')
cap('pt2.png', 'Table P1-2', 'Test results of 5-nearest neighbors on 143 tumors.', 'pt2s.png')
cap('pt3.png', 'Table P1-3', 'The first six patients of the test set.', 'pt3s.png')

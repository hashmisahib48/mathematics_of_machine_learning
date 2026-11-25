"""Sidebar figures for the four problems of Chapter 2 (real data, 1.75 in)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from PIL import Image
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_diabetes
W = 1.75; DPI = 300; CL = [BLUE, OR, RED]
def sty(ax, xl, yl):
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7.5, length=2.2)
def save(fig, name): fig.savefig(os.path.join(HERE, 'png', name), dpi=DPI, facecolor='white'); plt.close(fig)
def cap(src, label, text, out):
    im = Image.open(os.path.join(HERE, 'png', src)).convert('RGB'); w, h = im.size; ih = W * h / w
    probe = plt.figure(figsize=(W, 1), dpi=DPI); rr = probe.canvas.get_renderer()
    def width(s):
        t = probe.text(0, 0, s, fontsize=8, family='Liberation Serif')
        x = t.get_window_extent(renderer=rr).width / DPI; t.remove(); return x
    lw = width(label + '  '); lines, cur = [], ''
    for wd in text.split():
        trial = (cur + ' ' + wd).strip()
        if width(trial) + (lw if not lines else 0) <= W - 0.02 or not cur: cur = trial
        else: lines.append(cur); cur = wd
    lines.append(cur); plt.close(probe)
    lh = 8 / 72 * 1.2; ch = 0.06 + lh * len(lines)
    fig = plt.figure(figsize=(W, ih + ch), dpi=DPI); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(ih + ch, 0); ax.axis('off')
    ax.imshow(im, extent=(0, W, ih, 0), aspect='auto')
    y = ih + 0.06 + lh * 0.8
    ax.text(0, y, label, fontsize=8, family='Liberation Serif', weight='bold', color='#1F5FBF', va='baseline')
    for k, ln in enumerate(lines):
        ax.text(lw if k == 0 else 0, y + k * lh, ln, fontsize=8, family='Liberation Serif', color='#222222', va='baseline')
    fig.savefig(os.path.join(HERE, 'png', out), dpi=DPI, facecolor='white'); plt.close(fig)
# iris
I = load_iris(); fig, ax = plt.subplots(figsize=(W, 1.35), dpi=DPI)
for k in range(3):
    s = I.target == k; ax.plot(I.data[s, 2], I.data[s, 3], 'o', ms=2.2, color=CL[k], label=I.target_names[k])
ax.legend(fontsize=6.5, loc='upper left', handletextpad=0.1, borderaxespad=0.1, labelspacing=0.2)
sty(ax, 'petal length (cm)', 'petal width (cm)'); fig.tight_layout(pad=0.2); save(fig, 'pp1.png')
cap('pp1.png', 'Figure P2-1', 'Petal width against petal length for the 150 flowers.', 'pp1s.png')
# breast cancer
B = load_breast_cancer(); rad = B.data[:, 0]; fig, ax = plt.subplots(figsize=(W, 1.35), dpi=DPI)
bins = np.linspace(6, 29, 24)
ax.hist(rad[B.target == 1], bins=bins, color=BLUE, alpha=0.8, label='benign')
ax.hist(rad[B.target == 0], bins=bins, color=RED, alpha=0.8, label='malignant')
ax.axvline(15, color=NAVY, lw=0.9, ls='--'); ax.legend(fontsize=6.5, loc='upper right', borderaxespad=0.1, labelspacing=0.2)
sty(ax, 'mean radius', 'tumors'); fig.tight_layout(pad=0.2); save(fig, 'pp2.png')
cap('pp2.png', 'Figure P2-2', 'Mean radius of 569 tumors. The dashed line marks the radius 15.', 'pp2s.png')
# wine: raw and standardized on equal axes
Wn = load_wine(); X = Wn.data[:, [12, 10]]; Z = (X - X.mean(0)) / X.std(0, ddof=1)
fig, (a, b) = plt.subplots(2, 1, figsize=(W, 2.5), dpi=DPI)
a.plot(X[:, 0], X[:, 1], 'o', ms=1.8, color=BLUE); a.set_xlim(-100, 1800); a.set_ylim(-100, 1800)
a.set_aspect('equal'); sty(a, 'proline', 'hue'); a.set_xticks([0, 800, 1600]); a.set_yticks([0, 800, 1600])
b.plot(Z[:, 0], Z[:, 1], 'o', ms=1.8, color=GREEN); b.set_xlim(-3, 3.5); b.set_ylim(-3, 3.5)
b.set_aspect('equal'); sty(b, 'standardized proline', 'std. hue')
fig.tight_layout(pad=0.2, h_pad=0.5); save(fig, 'pp3.png')
cap('pp3.png', 'Figure P2-3', 'Hue against proline for 178 wines, in raw units (top) and standardized (bottom), on equal axes.', 'pp3s.png')
# diabetes
D = load_diabetes(); fig, ax = plt.subplots(figsize=(W, 1.35), dpi=DPI)
ax.plot(D.data[:, 2], D.target, 'o', ms=1.8, color=BLUE, alpha=0.7)
sty(ax, 'bmi (centered, scaled)', 'progression'); fig.tight_layout(pad=0.2); save(fig, 'pp4.png')
cap('pp4.png', 'Figure P2-4', 'Disease progression against bmi for 442 patients.', 'pp4s.png')

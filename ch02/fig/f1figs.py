"""Figures of Chapter 1 (sidebar width)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
from figstyle import *
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.datasets import load_iris
P = lambda n: os.path.join(HERE, 'png', n)
def fin(fig, n): fig.savefig(P(n), dpi=300, facecolor='white'); plt.close(fig)
def sty(ax, xl, yl): ax.set_xlabel(xl, fontsize=7); ax.set_ylabel(yl, fontsize=7); ax.tick_params(labelsize=6.3, length=2.2)
# 1.1-1 regression and classification
fig, (a, b) = plt.subplots(2, 1, figsize=(2.05, 3.0), dpi=300)
x = np.array([1, 2, 3, 4, 5.]); y = np.array([52, 57, 61, 68, 72.])
a.plot(x, y, 'o', ms=3.4, color=BLUE); xx = np.array([0.6, 5.4]); a.plot(xx, 5.1 * xx + 46.7, color=GREEN, lw=1.1)
a.set_title('regression: a number', fontsize=7, color=NAVY); sty(a, 'hours studied', 'test score')
I = load_iris(); X = I.data; t = I.target
for k, (c, m) in enumerate(zip([BLUE, OR, RED], ['o', 's', '^'])):
    b.plot(X[t == k, 2], X[t == k, 3], m, ms=2.4, color=c, label=I.target_names[k], alpha=0.85)
b.legend(fontsize=5.6, frameon=False, loc='upper left', handletextpad=0.2)
b.set_title('classification: a class', fontsize=7, color=NAVY); sty(b, 'petal length (cm)', 'petal width (cm)')
fig.tight_layout(pad=0.3, h_pad=0.7); fin(fig, 'f11.png')
# 1.3-1 the data table
fig = plt.figure(figsize=(2.05, 1.85), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(9, 0); ax.axis('off')
n, p = 6, 4; x0, y0, cw, rh = 1.2, 1.4, 1.25, 0.95
for i in range(n):
    for j in range(p):
        lab = '⋮' if i == 4 else (f'$x_{{{["1","2","3","4","","n"][i]}{["1","2","","p"][j]}}}$' if j != 2 else '⋯')
        if i == 4 and j == 2: lab = ''
        ax.add_patch(Rectangle((x0 + j * cw, y0 + i * rh), cw, rh, fc=LB if i % 2 == 0 else 'white', ec=BLUE, lw=0.5))
        ax.text(x0 + (j + 0.5) * cw, y0 + (i + 0.55) * rh, lab, ha='center', va='center', fontsize=7)
    ax.add_patch(Rectangle((x0 + p * cw + 0.45, y0 + i * rh), cw, rh, fc=PINK, ec=RED, lw=0.5))
    ax.text(x0 + p * cw + 0.45 + cw / 2, y0 + (i + 0.55) * rh, '⋮' if i == 4 else f'$y_{{{["1","2","3","4","","n"][i]}}}$', ha='center', va='center', fontsize=7)
ax.text(x0 + p * cw / 2, 0.85, 'features  $X$  ($n\\times p$)', ha='center', fontsize=7, color=BLUE, weight='bold')
ax.text(x0 + p * cw + 0.45 + cw / 2, 0.85, 'target $y$', ha='center', fontsize=7, color=RED, weight='bold')
ax.text(0.55, y0 + 3 * rh, 'one row per\nobservation', ha='center', va='center', fontsize=5.8, color=GREY, rotation=90)
fin(fig, 'f13.png')
# 1.4-1 underfitting and overfitting
d = np.load(os.path.join(HERE, '..', 'verify', 'overfit.npz')); x, y = d['x'], d['y']
fig, ax = plt.subplots(figsize=(2.05, 1.75), dpi=300)
xx = np.linspace(x.min(), x.max(), 400)
ax.plot(x, y, 'o', ms=3.2, color=NAVY, zorder=5, label='training data')
for deg, c, ls in [(1, OR, '-'), (3, GREEN, '-'), (9, RED, '-')]:
    ax.plot(xx, np.polyval(np.polyfit(x, y, deg), xx), color=c, lw=1.0, ls=ls, label=f'degree {deg}')
ax.set_ylim(-0.9, 3.3); ax.legend(fontsize=5.5, frameon=False, loc='upper left', ncol=2, handlelength=1.4, columnspacing=0.8)
sty(ax, '$x$', '$y$'); fig.tight_layout(pad=0.3); fin(fig, 'f14.png')
# 1.4-2 workflow
fig = plt.figure(figsize=(2.05, 2.3), dpi=300); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(11, 0); ax.axis('off')
def bx(x, y, w, h, t, fc, ec): ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.25', fc=fc, ec=ec, lw=0.8)); ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=6.6, color=NAVY)
bx(2.5, 0.3, 5, 1.3, 'data  $X$, $y$', LGREY, GREY)
bx(0.3, 3.0, 4.2, 1.5, 'training set\n75 %', LB, BLUE); bx(5.5, 3.0, 4.2, 1.5, 'test set\n25 %', PINK, RED)
bx(0.3, 5.9, 4.2, 1.5, 'fit\nlearn the weights', LB, BLUE); bx(5.5, 5.9, 4.2, 1.5, 'predict\n$\\hat y$', PINK, RED)
bx(2.5, 8.8, 5, 1.5, 'score\ncompare $\\hat y$ with $y$', '#E7F4EC', GREEN)
for p, q in [((4.2, 1.6), (2.4, 3.0)), ((5.8, 1.6), (7.6, 3.0)), ((2.4, 4.5), (2.4, 5.9)), ((7.6, 4.5), (7.6, 5.9)), ((4.5, 6.65), (5.5, 6.65)), ((7.6, 7.4), (6.4, 8.8))]:
    arr(ax, p, q, c=GREY, lw=0.7, ms=5)

fin(fig, 'f15.png')
print('ok')

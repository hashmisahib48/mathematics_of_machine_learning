# Figures of Chapter 33, part 1: schematics and the six-point example. Run: python3 fig/figs1.py
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Rectangle
from sklearn.datasets import load_linnerud
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CL = [BLUE, OR, TEAL]; MK = ['o', 's', '^']

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

x = np.array([1., 2, 4, 5, 7, 8]); y = np.array([0, 0, 1, 1, 2, 2])

# ---------------------------------------------------------------- f330 : three kinds of target
d = load_linnerud(); Yl = d.target[:5].astype(int)
fig, ax = new(5.7, 2.35)
def table(x0, y0, rows, cw, ch, heads, fills, fs=9.5, hfs=9.5):
    for j, h in enumerate(heads):
        ax.text(x0 + (j + .5) * cw, y0 + 0.13, h, ha='center', va='bottom', fontsize=hfs, color=NAVY)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            ax.add_patch(Rectangle((x0 + j * cw, y0 - (i + 1) * ch), cw, ch, fc=fills[i][j], ec='white', lw=1.2))
            ax.text(x0 + (j + .5) * cw, y0 - (i + .5) * ch, str(v), ha='center', va='center', fontsize=fs, color=NAVY)
lab = [2, 0, 1, 2, 1]
table(0.0, 2.0, [[v] for v in lab], 0.75, 0.30, ['bin'], [[[LB, '#FDF0E0', '#DDF0F1'][v]] for v in lab])
M = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]])
table(1.75, 2.0, M.tolist(), 0.52, 0.30, ['sea', 'sky', 'dog'], [['#FDF0E0' if v else LGREY for v in r] for r in M])
table(3.75, 2.0, Yl.tolist(), 0.62, 0.30, ['weight', 'waist', 'pulse'], [['#E3F4EA'] * 3 for _ in Yl])
for xc, t1, t2, t3 in [(0.375, 'multiclass', 'one label of K', 'OneVsRest, OneVsOne,\nOutputCode'),
                       (2.53, 'multilabel', 'any subset of K labels', 'MultiOutputClassifier,\nClassifierChain'),
                       (4.68, 'multioutput', 'several numbers', 'MultiOutputRegressor,\nRegressorChain')]:
    ax.text(xc, 0.33, t1, ha='center', fontsize=10.5, color=BLUE, weight='bold')
    ax.text(xc, 0.07, t2, ha='center', fontsize=9.5, color=GREY)
    ax.text(xc, -0.17, t3, ha='center', va='top', fontsize=9, color=PUR, linespacing=1.25)
ax.set_xlim(-0.45, 5.95); ax.set_ylim(-0.62, 2.45)
sv(fig, 'f330.png', 470)

# ---------------------------------------------------------------- f331 : the six points
fig, ax = plt.subplots(figsize=(4.4, 1.55), dpi=240)
for k in range(3):
    m = y == k; ax.scatter(x[m], np.zeros(m.sum()), s=60, color=CL[k], marker=MK[k], edgecolor='white', lw=0.8, zorder=5)
    ax.text(x[m].mean(), 0.42, f'class {k}', ha='center', fontsize=9.5, color=CL[k], weight='bold')
for xi in x: ax.text(xi, -0.55, f'{xi:.0f}', ha='center', fontsize=9, color=GREY)
ax.axhline(0, color=GREY, lw=0.8, zorder=1); ax.set_ylim(-0.9, 0.9); ax.set_xlim(0, 9); ax.axis('off')
ax.text(4.5, -0.95, 'feature  $x$', ha='center', fontsize=9.5, color='#222222', va='top')
sv(fig, 'f331.png', 400)

# ---------------------------------------------------------------- f332 : every scheme is a code book
M_ecoc = np.array([[1, 1, 1, 1, 1, 1], [1, 1, -1, -1, -1, -1], [-1, -1, 1, 1, -1, -1], [-1, -1, -1, -1, 1, 1]])
ovr = -np.ones((4, 4), int) + 2 * np.eye(4, dtype=int)
pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]; ovo = np.zeros((4, 6), int)
for c, (j, k) in enumerate(pairs): ovo[j, c] = -1; ovo[k, c] = 1
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.15), dpi=240, gridspec_kw={'width_ratios': [4, 6, 6], 'wspace': 0.22})
for ax, Mx, t in zip(axs, [ovr, ovo, M_ecoc], ['one-versus-rest\n$K$ = 4 problems', 'one-versus-one\n$K(K-1)/2$ = 6 problems', 'output code\n$B$ = 6 bits']):
    n, m = Mx.shape
    for i in range(n):
        for j in range(m):
            v = Mx[i, j]; fc = {1: '#FDF0E0', -1: LB, 0: '#F2F3F5'}[v]; ec = {1: OR, -1: BLUE, 0: GRID}[v]
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, fc=fc, ec='white', lw=1.5))
            ax.text(j + .5, n - 1 - i + .5, {1: '+', -1: '−', 0: '0'}[v], ha='center', va='center', fontsize=10, color={1: OR, -1: BLUE, 0: '#9AA0A8'}[v], weight='bold')
    ax.set_xlim(0, m); ax.set_ylim(0, n); ax.axis('off'); ax.set_title(t, fontsize=9.5, color=NAVY, pad=3)
    for j in range(m): ax.text(j + .5, -0.2, str(j + 1), ha='center', va='top', fontsize=9, color=GREY)
    ax.text(m / 2, -0.85, 'problem', ha='center', va='top', fontsize=9, color=GREY)
for i, c_ in enumerate('ABCD'): axs[0].text(-0.2, 3 - i + .5, c_, ha='right', va='center', fontsize=10, color=NAVY, weight='bold')
axs[0].text(-0.2, 4.35, 'class', ha='right', va='bottom', fontsize=9, color=GREY)
sv(fig, 'f332.png', 470)

# ---------------------------------------------------------------- f341 : independent labels and a chain
fig, ax = new(5.6, 2.1)
def bx(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=9.5): box(ax, xc - w / 2, yc - h / 2, w, h, t, fc=fc, ec=ec, fs=fs, r=0.06)
A = lambda p, q, c=NAVY: arr(ax, p, q, c=c, lw=0.9, ms=8)
ax.text(1.35, 2.05, 'independent  (MultiOutputClassifier)', ha='center', fontsize=9.5, color=NAVY, weight='bold')
bx(0.25, 1.0, 0.5, 0.45, '$\\mathbf{x}$', fc=LGREY, ec=GREY)
for k, yy in enumerate([1.7, 1.0, 0.3]):
    bx(2.1, yy, 0.62, 0.42, f'$\\hat y_{k + 1}$', fc='#E3F4EA', ec=GREEN); A((0.5, 1.0 + (yy - 1.0) * 0.25), (1.79, yy))
ax.text(4.95, 2.05, 'chain  (ClassifierChain)', ha='center', fontsize=9.5, color=NAVY, weight='bold')
bx(3.55, 1.0, 0.5, 0.45, '$\\mathbf{x}$', fc=LGREY, ec=GREY)
xs = [4.55, 5.55, 6.55]
for k, xc in enumerate(xs):
    bx(xc, 1.0, 0.62, 0.42, f'$\\hat y_{k + 1}$', fc='#E3F4EA', ec=GREEN)
A((3.8, 1.0), (4.24, 1.0)); A((4.86, 1.0), (5.24, 1.0)); A((5.86, 1.0), (6.24, 1.0))
for xc in xs[1:]:
    pass
ax.annotate('', xy=(5.55, 0.78), xytext=(3.55, 0.77), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.8, connectionstyle='arc3,rad=0.30', mutation_scale=7))
ax.annotate('', xy=(6.55, 0.78), xytext=(3.55, 0.77), arrowprops=dict(arrowstyle='-|>', color=GREY, lw=0.8, connectionstyle='arc3,rad=0.38', mutation_scale=7))
ax.text(5.05, -0.05, '$\\mathbf{x}$ goes to every link; link $l$ also sees $\\hat y_1,\\dots,\\hat y_{l-1}$', ha='center', va='center', fontsize=9, color=GREY)
ax.set_xlim(-0.1, 7.1); ax.set_ylim(-0.3, 2.3)
sv(fig, 'f341.png', 470)

# ---------------------------------------------------------------- f351 : the six points, three score lines, three pair lines
xx = np.linspace(0.3, 8.7, 200)
def line(xs, zs):
    w = ((xs - xs.mean()) * (zs - zs.mean())).sum() / ((xs - xs.mean()) ** 2).sum(); return w, zs.mean() - w * xs.mean()
S = [line(x, (y == k) * 1.0) for k in range(3)]
plt.rcParams.update({'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5, 'axes.labelsize': 10})
fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.6), dpi=240)
ax = axs[0]
for k in range(3):
    ax.plot(xx, S[k][0] * xx + S[k][1], color=CL[k], lw=1.8, ls=['-', '--', ':'][k], zorder=3)
ax.axvspan(0.3, 4.5, color='#E3EEFB', zorder=0); ax.axvspan(4.5, 8.7, color='#DDF0F1', zorder=0)
for k in range(3):
    m = y == k; ax.scatter(x[m], np.full(m.sum(), -0.12), s=40, color=CL[k], marker=MK[k], edgecolor='white', lw=0.7, zorder=6, clip_on=False)
ax.text(2.4, 1.03, 'class 0 wins', ha='center', fontsize=9.5, color=BLUE); ax.text(6.6, 1.03, 'class 2 wins', ha='center', fontsize=9.5, color=TEAL)
ax.text(8.6, 0.37, '$s_1$ (flat)', ha='right', va='bottom', fontsize=9.5, color=OR, weight='bold')
ax.text(0.5, 0.66, '$s_0$', fontsize=10, color=BLUE, weight='bold'); ax.text(7.9, 0.02, '$s_2$', fontsize=10, color=TEAL, weight='bold')
ax.set_xlim(0.3, 8.7); ax.set_ylim(-0.2, 1.12); ax.set_xlabel('$x$'); ax.set_ylabel('score $s_k(x)$'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title('one-versus-rest', fontsize=10, color=NAVY)
ax = axs[1]
PR = [(0, 1), (1, 2), (0, 2)]; pc = [PUR, GREEN, RED]
for (j, k), c_ in zip(PR, pc):
    m = (y == j) | (y == k); w, b = line(x[m], np.where(y[m] == k, 1.0, -1.0))
    ax.plot(xx, w * xx + b, color=c_, lw=1.6, ls=['-', '--', ':'][PR.index((j, k))], zorder=3)
ax.axhline(0, color=GREY, lw=0.8)
ax.text(0.5, 1.25, '$f_{01}$', color=PUR, fontsize=10, weight='bold'); ax.text(5.4, -1.45, '$f_{12}$', color=GREEN, fontsize=10, weight='bold', ha='left')
ax.text(8.55, 2.45, '$f_{02}$', color=RED, fontsize=10, weight='bold', ha='right')
for k in range(3):
    m = y == k; ax.scatter(x[m], np.full(m.sum(), -2.95), s=40, color=CL[k], marker=MK[k], edgecolor='white', lw=0.7, zorder=6, clip_on=False)
ax.set_xlim(0.3, 8.7); ax.set_ylim(-3.3, 3.4); ax.set_xlabel('$x$'); ax.set_ylabel('pair score $f_{jk}(x)$'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title('one-versus-one', fontsize=10, color=NAVY)
fig.tight_layout(w_pad=1.0)
sv(fig, 'f351.png', 470)

# ---------------------------------------------------------------- f361 : flow chart of the three wrappers (units = inches)
S = 0.78   # vertical compression of the first layout
fig = plt.figure(figsize=(4.9, 4.45 * S + 0.04), dpi=240); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 4.9); ax.set_ylim(-0.04, 4.45 * S); ax.axis('off')
def fb(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=9): box(ax, xc - w / 2, S * yc - S * h / 2, w, S * h, t, fc=fc, ec=ec, fs=fs, r=0.05)
def ln(xs, ys): ax.plot(xs, [S * v for v in ys], color=NAVY, lw=0.9)
A = lambda p, q, **k: arr(ax, (p[0], S * p[1]), (q[0], S * q[1]), c=NAVY, lw=0.9, ms=8, **k)
C3 = (0.85, 2.45, 4.05); BW = 1.52
fb(2.45, 4.2, 4.7, 0.4, 'fit($\\mathbf{X}$, $y$):  find classes_;  $K$ = their number', fc=LGREY, ec=GREY)
ln([2.45, 2.45], [4.0, 3.9]); ln([C3[0], C3[2]], [3.9, 3.9])
for xc in C3: A((xc, 3.9), (xc, 3.78))
fb(C3[0], 3.3, BW, 0.95, 'OneVsRest\n$K$ problems: class $k$\nagainst the rest', fc='#E3F4EA', ec=GREEN, fs=8.8)
fb(C3[1], 3.3, BW, 0.95, 'OneVsOne\n$K(K-1)/2$ problems:\none pair of classes', fc='#E3F4EA', ec=GREEN, fs=8.8)
fb(C3[2], 3.3, BW, 0.95, 'OutputCode\n$B$ problems: one bit\nof a random code', fc='#E3F4EA', ec=GREEN, fs=8.8)
for xc in C3: ln([xc, xc], [2.82, 2.72])
ln([C3[0], C3[2]], [2.72, 2.72]); A((2.45, 2.72), (2.45, 2.62))
fb(2.45, 2.42, 4.7, 0.4, 'clone the base estimator and fit it on each problem', fc=LB, ec=BLUE)
A((2.45, 2.22), (2.45, 2.0))
fb(2.45, 1.8, 4.7, 0.4, 'predict($\\mathbf{X}$):  score every row with every estimator', fc='#F1EAF7', ec=PUR)
ln([2.45, 2.45], [1.6, 1.5]); ln([C3[0], C3[2]], [1.5, 1.5])
for xc in C3: A((xc, 1.5), (xc, 1.38))
fb(C3[0], 1.0, BW, 0.74, 'largest score\n$\\arg\\max_k\\, f_k$', fc='#F1EAF7', ec=PUR, fs=8.8)
fb(C3[1], 1.0, BW, 0.74, 'most votes; ties by\nsummed confidence', fc='#F1EAF7', ec=PUR, fs=8.8)
fb(C3[2], 1.0, BW, 0.74, 'nearest code word\nto the score vector', fc='#F1EAF7', ec=PUR, fs=8.8)
for xc in C3: ln([xc, xc], [0.63, 0.53])
ln([C3[0], C3[2]], [0.53, 0.53]); A((2.45, 0.53), (2.45, 0.42))
fb(2.45, 0.22, 4.7, 0.4, 'return classes_[ index of the winner ]', fc=LGREY, ec=GREY)
sv(fig, 'f361.png', 450)

json.dump(NUM, open('verify/fig_numbers1.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize1.json', 'w'), indent=1)
print(json.dumps(CHK, indent=0))

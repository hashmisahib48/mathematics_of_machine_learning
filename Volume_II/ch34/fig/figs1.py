# Figures of Chapter 34, part 1: opening, label flow, chain, flow chart. Run: python3 fig/figs1.py
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.semi_supervised import LabelPropagation
warnings.simplefilter('ignore')
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CL = [BLUE, OR]; MK = ['o', 's']
CM = LinearSegmentedColormap.from_list('bo', [BLUE, '#F2F2F2', OR])

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

# ---------------- the two-moons data used in the opening figures
X, y = make_moons(200, noise=0.1, random_state=1)
i0 = np.where(y == 0)[0][np.argmin(X[y == 0, 0])]; i1 = np.where(y == 1)[0][np.argmax(X[y == 1, 0])]
lab = np.array([i0, i1]); yy = -np.ones(200, int); yy[lab] = y[lab]
unl = np.setdiff1d(np.arange(200), lab)

def frame(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.set_xlim(-1.45, 2.5); ax.set_ylim(-0.95, 1.4)
    for s in ax.spines.values(): s.set_visible(False)
    if title: ax.set_title(title, fontsize=9.5, color=NAVY, pad=2)

def labelled(ax):
    for j, i in enumerate(lab):
        ax.scatter(X[i, 0], X[i, 1], s=70, marker=MK[y[i]], color=CL[y[i]], edgecolor=NAVY, lw=1.1, zorder=6)

# ---------------------------------------------------------------- f340 : the chapter in one picture
fig, axs = plt.subplots(1, 3, figsize=(5.7, 1.95), dpi=240)
ax = axs[0]; frame(ax, 'two labels, no more')
ax.scatter(X[unl, 0], X[unl, 1], s=9, color='#B8BEC8', lw=0, zorder=2)
sup = LogisticRegression().fit(X[lab], y[lab]); w = sup.coef_[0]; b = sup.intercept_[0]
xs = np.array([-1.45, 2.5]); ys = -(w[0] * xs + b) / w[1]
ax.plot(xs, ys, color=GREEN, lw=1.5, ls='--', zorder=3)
ax.text(2.45, 1.38, 'boundary from the\nlabels alone', ha='right', va='top', fontsize=9.2, color=GREEN)
labelled(ax)
NUM['sup_acc_moons'] = float(sup.score(X, y))
ax = axs[1]; frame(ax, 'a graph on all samples')
nn = NearestNeighbors(n_neighbors=6).fit(X); idx = nn.kneighbors(X, return_distance=False)[:, 1:]
segs = [[X[i], X[j]] for i in range(200) for j in idx[i]]
ax.add_collection(LineCollection(segs, colors='#9AA3B0', linewidths=0.35, zorder=1))
ax.scatter(X[unl, 0], X[unl, 1], s=9, color='#B8BEC8', lw=0, zorder=2); labelled(ax)
ax = axs[2]; frame(ax, 'labels flow along the graph')
lp = LabelPropagation(gamma=50, max_iter=200000, tol=1e-9).fit(X, yy); tr = lp.transduction_
for k in (0, 1):
    m = (tr == k); m[lab] = False
    ax.scatter(X[m, 0], X[m, 1], s=11, marker=MK[k], color=CL[k], lw=0, zorder=2)
labelled(ax)
NUM['lp_acc_moons'] = float((tr == y).mean()); NUM['lp_n_iter'] = int(lp.n_iter_)
fig.subplots_adjust(wspace=0.04, left=0.01, right=0.99, top=0.9, bottom=0.01)
sv(fig, 'f340.png', 470)

# ---------------------------------------------------------------- f342 : the label flow, iteration by iteration
steps = [1, 100, 2000, 25000]
fig, axs = plt.subplots(1, 4, figsize=(5.7, 1.62), dpi=240)
FL = {}
for ax, t in zip(axs, steps):
    m = LabelPropagation(gamma=50, max_iter=t, tol=0).fit(X, yy)
    P = m.label_distributions_[:, 1]; FL[t] = float(((P > 0.5) == y).mean())
    frame(ax, f'$t = {t}$')
    ax.scatter(X[unl, 0], X[unl, 1], s=12, c=P[unl], cmap=CM, vmin=0, vmax=1, lw=0, zorder=2)
    labelled(ax)
    ax.text(0.5, -0.06, f'accuracy {FL[t]:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=9, color=GREY)
NUM['flow_accuracy'] = FL
NUM['moon_final_P'] = {'class0_mean': float(P[unl][y[unl] == 0].mean()), 'class1_mean': float(P[unl][y[unl] == 1].mean()), 'min': float(P[unl].min()), 'max': float(P[unl].max())}
fig.subplots_adjust(wspace=0.03, left=0.005, right=0.995, top=0.9, bottom=0.08)
sv(fig, 'f342.png', 470)

# ---------------------------------------------------------------- f351 : the chain
Wc = np.zeros((5, 5))
for i, w_ in enumerate([2, 4, 1, 1]): Wc[i, i + 1] = Wc[i + 1, i] = w_
dc = Wc.sum(1); Lc, Uc = [0, 4], [1, 2, 3]; Tc = Wc / dc[:, None]
fU = np.linalg.solve(np.diag(dc[Uc]) - Wc[np.ix_(Uc, Uc)], Wc[np.ix_(Uc, Lc)] @ np.array([0., 1.]))
NUM['chain_fU'] = fU.tolist()
fig, axs = plt.subplots(1, 2, figsize=(5.7, 2.35), dpi=240, gridspec_kw={'width_ratios': [1.05, 1]})
ax = axs[0]; ax.axis('off'); ax.set_xlim(-0.5, 4.5); ax.set_ylim(-1.35, 1.35); ax.set_aspect('equal')
for i, w_ in enumerate([2, 4, 1, 1]):
    ax.plot([i, i + 1], [0, 0], color=GREY, lw=0.8 + 1.15 * w_, zorder=1, solid_capstyle='butt')
    ax.text(i + 0.5, 0.32, f'{w_}', ha='center', fontsize=10, color=NAVY)
ax.text(2.0, 0.92, 'similarity $w_{ij}$ of neighbours', ha='center', fontsize=9, color=GREY)
fv = [0, fU[0], fU[1], fU[2], 1]
for i in range(5):
    if i in Lc:
        k = 0 if i == 0 else 1
        ax.scatter(i, 0, s=330, marker=MK[k], color=CL[k], edgecolor=NAVY, lw=1.2, zorder=5)
    else:
        ax.scatter(i, 0, s=330, marker='o', color=CM(fv[i]), edgecolor=NAVY, lw=1.0, zorder=5)
    ax.text(i, 0, f'{i + 1}', ha='center', va='center', fontsize=10, color='white' if i in Lc else NAVY, weight='bold', zorder=6)
    ax.text(i, -0.52, ['0', '2/11', '3/11', '7/11', '1'][i], ha='center', va='top', fontsize=9.5, color=GREEN if i in Uc else NAVY)
ax.text(2.0, -1.2, 'harmonic value $f_i$ = share of class 1', ha='center', fontsize=9, color=GREEN)
ax = axs[1]
g = np.array([0, .5, .5, .5, 1.0]); g[Lc] = [0, 1]; hist = {0: g.copy()}
for t in range(1, 11):
    g = Tc @ g; g[Lc] = [0, 1]; hist[t] = g.copy()
xs = np.arange(1, 6)
for t, st, c_, lw in [(0, ':', GREY, 1.0), (1, '--', PUR, 1.2), (3, '-.', PUR, 1.2), (10, '-', PUR, 1.4)]:
    ax.plot(xs, hist[t], ls=st, color=c_, lw=lw, marker='o', ms=3.2, label=f'$t = {t}$', zorder=3)
ax.plot(xs, [0, fU[0], fU[1], fU[2], 1], color=GREEN, lw=2.0, label='limit', zorder=4)
ax.set_xticks(xs); ax.set_xlabel('node $i$'); ax.set_ylabel('$f_i$'); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='upper left', fontsize=8.8, handlelength=1.8, labelspacing=0.2, borderpad=0.1)
ax.set_ylim(-0.04, 1.04)
fig.tight_layout(w_pad=1.2)
sv(fig, 'f351.png', 470)
NUM['chain_iter'] = {str(t): hist[t].round(4).tolist() for t in (0, 1, 3, 10)}

# ---------------------------------------------------------------- f361 : flow chart
fig = plt.figure(figsize=(4.9, 4.3), dpi=240); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 4.9); ax.set_ylim(0, 4.3); ax.axis('off')
def fb(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=8.8): box(ax, xc - w / 2, yc - h / 2, w, h, t, fc=fc, ec=ec, fs=fs, r=0.05)
A_ = lambda p, q, **k: arr(ax, p, q, c=NAVY, lw=0.9, ms=8, **k)
XL, XR, BW = 1.28, 3.52, 2.14
ax.text(XL, 4.14, 'graph methods', ha='center', fontsize=10, color=BLUE, weight='bold')
ax.text(XR, 4.14, 'self-training', ha='center', fontsize=10, color=BLUE, weight='bold')
fb(XL, 3.65, BW, 0.5, 'build $\\mathbf{W}$ from $\\mathbf{X}$:\nrbf or k nearest neighbours', fc=LGREY, ec=GREY)
fb(XL, 2.9, BW, 0.55, '$\\mathbf{T}=\\mathbf{D}^{-1}\\mathbf{W}$  or\n$\\mathbf{S}=\\mathbf{D}^{-1/2}\\mathbf{W}\\mathbf{D}^{-1/2}$', fc='#E3F4EA', ec=GREEN)
fb(XL, 2.15, BW, 0.55, '$\\mathbf{F}\\leftarrow\\mathbf{T}\\mathbf{F}$, clamp labelled rows\nor  $\\mathbf{F}\\leftarrow\\alpha\\mathbf{S}\\mathbf{F}+(1-\\alpha)\\mathbf{Y}$', fc='#F1EAF7', ec=PUR)
fb(XL, 1.4, BW, 0.45, 'change below tol,\nor max_iter reached?', fc='#FDF0E0', ec=OR)
fb(XL, 0.5, BW, 0.55, 'normalise rows: label_distributions_\ntransduction_ = arg max', fc=LGREY, ec=GREY)
A_((XL, 3.4), (XL, 3.19)); A_((XL, 2.62), (XL, 2.44)); A_((XL, 1.87), (XL, 1.63)); A_((XL, 1.17), (XL, 0.79))
ax.text(XL + 0.06, 1.0, 'yes', fontsize=8.5, color=NAVY, va='center')
ax.plot([XL - BW / 2, XL - BW / 2 - 0.12, XL - BW / 2 - 0.12], [1.4, 1.4, 2.15], color=NAVY, lw=0.9); A_((XL - BW / 2 - 0.12, 2.15), (XL - BW / 2, 2.15))
ax.text(XL - BW / 2 - 0.15, 1.78, 'no', fontsize=8.5, color=NAVY, rotation=90, ha='right', va='center')
fb(XR, 3.65, BW, 0.5, 'fit the base estimator on\nthe labelled samples', fc=LB, ec=BLUE)
fb(XR, 2.9, BW, 0.55, 'predict_proba on the\nunlabelled samples', fc=LB, ec=BLUE)
fb(XR, 2.15, BW, 0.55, 'keep the confident ones:\nmax prob > threshold, or k_best', fc='#E3F4EA', ec=GREEN)
fb(XR, 1.4, BW, 0.45, 'none added, max_iter\nreached, or all labelled?', fc='#FDF0E0', ec=OR)
fb(XR, 0.5, BW, 0.55, 'final fit on labelled and\npseudo-labelled samples', fc=LGREY, ec=GREY)
A_((XR, 3.4), (XR, 3.19)); A_((XR, 2.62), (XR, 2.44)); A_((XR, 1.87), (XR, 1.63)); A_((XR, 1.17), (XR, 0.79))
ax.text(XR + 0.06, 1.0, 'yes', fontsize=8.5, color=NAVY, va='center')
ax.plot([XR + BW / 2, XR + BW / 2 + 0.12, XR + BW / 2 + 0.12], [1.4, 1.4, 3.65], color=NAVY, lw=0.9); A_((XR + BW / 2 + 0.12, 3.65), (XR + BW / 2, 3.65))
ax.text(XR + BW / 2 + 0.17, 2.5, 'no', fontsize=8.5, color=NAVY, rotation=90, ha='left', va='center')
sv(fig, 'f361.png', 450)

json.dump(NUM, open('verify/fig_numbers1.json', 'w'), indent=1); json.dump(CHK, open('verify/fig_textsize1.json', 'w'), indent=1)
print(json.dumps(NUM)); print(json.dumps(CHK, indent=0))

# Figures 40.7-1, 40.9-1, 40.13-1, 40.13-2  (run: python3 fig/figs5.py)
from common import *
from matplotlib.patches import Polygon, FancyBboxPatch
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
NUM = {}
X, g = people_2d(); xlim = (100, 205); ylim = (10, 112)

# ---- 40.7-1: components switched off by a small concentration prior
fig, axs = plt.subplots(2, 3, figsize=(5.4, 3.0), dpi=240, gridspec_kw={'height_ratios': [1.25, 1]})
NUM['bgm'] = {}
for j, gam0 in enumerate([0.001, 10.0, 1000.0]):
    b = BGM(n_components=8, weight_concentration_prior_type='dirichlet_distribution', weight_concentration_prior=gam0, max_iter=500, tol=1e-6, random_state=0).fit(X)
    ax = axs[0, j]; ax.scatter(X[:, 0], X[:, 1], s=5, color=GRID, lw=0, zorder=2)
    act = b.weights_ > 0.01
    for k in np.where(act)[0]: ell(ax, b.means_[k], b.covariances_[k], nstd=2, color=GREEN, lw=1.2)
    ax.scatter(b.means_[act, 0], b.means_[act, 1], s=12, marker='+', color=NAVY, lw=0.9, zorder=6)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_xticks([120, 160, 200]); ax.set_xlabel('height', fontsize=9)
    ax.set_title(f'$\\gamma_0$ = {gam0:g}:  {int(act.sum())} active', fontsize=9, color=NAVY)
    if j == 0: ax.set_ylabel('weight'); ax.set_yticks([20, 60, 100])
    else: ax.set_yticklabels([])
    ax = axs[1, j]; w = np.sort(b.weights_)[::-1]
    ax.bar(np.arange(1, 9), w, color=[GREEN if v > 0.01 else GRID for v in w], width=0.7); ax.set_ylim(0, 0.45); ax.set_xticks([1, 4, 8])
    ax.set_xlabel('component, sorted', fontsize=9)
    if j == 0: ax.set_ylabel('weight $\\pi_k$'); ax.set_yticks([0, 0.2, 0.4])
    else: ax.set_yticklabels([])
    NUM['bgm'][str(gam0)] = {'active': int(act.sum()), 'weights': w.round(3).tolist(), 'n_iter': int(b.n_iter_)}
fig.tight_layout(w_pad=0.4, h_pad=0.6); sv(fig, 'f4123.png', 470)

# ---- 40.9-1: flow chart of GaussianMixture.fit (wide layout, two rows)
fig, ax = new(5.4, 3.0)
def fb(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=9): box(ax, xc - w / 2, yc - h / 2, w, h, t, fc=fc, ec=ec, fs=fs, r=0.06)
A = lambda p, q, **k: arr(ax, p, q, c=NAVY, lw=0.9, ms=8, **k)
def dia(xc, yc, w, h, t, fs=9): ax.add_patch(Polygon([(xc, yc + h / 2), (xc + w / 2, yc), (xc, yc - h / 2), (xc - w / 2, yc)], fc='#FDF0E0', ec=OR, lw=0.9)); ax.text(xc, yc, t, ha='center', va='center', fontsize=fs, color=NAVY, linespacing=1.1)
def yn(x, y, t, ha='center'): ax.text(x, y, t, fontsize=9, color=GREY, style='italic', ha=ha, va='center')
X1, X2, X3, Y1, Y2 = 0.9, 2.7, 4.5, 2.25, 0.7
fb(X1, Y1, 1.6, 1.2, 'Input $\\mathbf{X}$, $K$\nStart $s$ = 1 ... n_init:\n$\\gamma$ from init_params,\nthen one M-step', fc='#F1EAF7', ec=PUR)
fb(X2, Y1, 1.6, 1.2, 'E-step: $\\ln\\,\\pi_k\\mathcal{N}$,\nlog-sum-exp, $\\gamma_{ik}$;\nbound = mean\n$\\ln\\,p(\\mathbf{x}_i)$')
fb(X3, Y1, 1.6, 1.2, 'M-step: $\\pi_k$, $\\boldsymbol{\\mu}_k$,\n$\\boldsymbol{\\Sigma}_k$ from $\\gamma_{ik}$;\nreg_covar added\nto the diagonal')
dia(X3, Y2, 1.7, 1.2, 'gain of bound\n< tol or\nmax_iter?')
dia(X2, Y2, 1.7, 1.2, 'last of the\nn_init starts?')
fb(X1, Y2, 1.6, 1.0, 'Keep the start with\nthe largest bound;\nfinal E-step', fc='#E3F4EA', ec=GREEN)
A((X1 + 0.8, Y1), (X2 - 0.8, Y1)); A((X2 + 0.8, Y1), (X3 - 0.8, Y1)); A((X3, Y1 - 0.6), (X3, Y2 + 0.6))
A((X3 - 0.85, Y2), (X2 + 0.85, Y2)); A((X2 - 0.85, Y2), (X1 + 0.8, Y2)); yn(3.6, Y2 - 0.22, 'yes'); yn(1.78, Y2 - 0.22, 'yes')
A((X3 - 0.55, Y2 + 0.35), (X2 + 0.5, Y1 - 0.6)); yn(3.75, 1.5, 'no')
A((X2 - 0.55, Y2 + 0.35), (X1 + 0.5, Y1 - 0.6)); yn(1.95, 1.5, 'no')
ax.set_xlim(0, 5.4); ax.set_ylim(0, 3.0); sv(fig, 'f491.png', 470)

# ---- 40.13-1: wine
Xw, yw = load_wine(return_X_y=True); Xw = StandardScaler().fit_transform(Xw)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4), dpi=240, gridspec_kw={'width_ratios': [1, 1.05]})
ax = axs[0]; sty = {'full': (GREEN, '-', 'o'), 'tied': (PUR, '--', 's'), 'diag': (OR, '-.', '^'), 'spherical': (BLUE, ':', 'v')}
NUM['wine_bic'] = {}; fits = {}
for ct in ['full', 'tied', 'diag', 'spherical']:
    row = []
    for k in range(1, 7):
        m = GM(k, covariance_type=ct, n_init=5, tol=1e-5, max_iter=500, random_state=0).fit(Xw); fits[ct, k] = m; row.append(m.bic(Xw))
    NUM['wine_bic'][ct] = [round(float(v), 1) for v in row]
    c_, ls_, mk_ = sty[ct]; ax.plot(range(1, 7), row, color=c_, ls=ls_, marker=mk_, ms=3.5, lw=1.3, label=ct)
    j = int(np.argmin(row)); ax.plot([j + 1], [row[j]], marker='*', color=c_, ms=11, mec='white', zorder=6)
ax.set_xlabel('number of components $K$'); ax.set_ylabel('BIC (lower is better)'); ax.set_xticks(range(1, 7)); ax.set_ylim(5400, 6450); grid(ax)
ax.legend(loc='upper right', fontsize=8.5, handlelength=2.0, borderaxespad=0.1, labelspacing=0.2)
ax = axs[1]; pc = PCA(2, random_state=0).fit(Xw); Zw = pc.transform(Xw); m = fits['diag', 3]; lab = m.predict(Xw)
order = np.argsort([m.means_[k] @ pc.components_[0] for k in range(3)])     # name components by their position along PC 1
cmap = {int(k): GCOL[i] for i, k in enumerate(order)}
for c in range(3):
    mk = ax.scatter(Zw[yw == c, 0], Zw[yw == c, 1], s=14, marker=GMK[c], c=[cmap[int(l)] for l in lab[yw == c]], lw=0, zorder=3)
P2 = pc.components_
for k in range(3):
    C_ = P2 @ np.diag(m.covariances_[k]) @ P2.T; ell(ax, P2 @ (m.means_[k]), C_, nstd=2, color=GREEN, lw=1.3)
ax.set_xlabel('first principal component'); ax.set_ylabel('second'); ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 3.9)
for c, nm in enumerate(['cultivar 0', 'cultivar 1', 'cultivar 2']): ax.scatter([], [], s=16, marker=GMK[c], color=GREY, label=nm)
ax.legend(loc='lower right', fontsize=8.3, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.15)
fig.tight_layout(w_pad=1.0); sv(fig, 'f4131.png', 470)
NUM['wine_diag3_ellipse_note'] = 'ellipses: 2-sd contour of the diag, K=3 model projected on the first two principal components'

# ---- 40.13-2: digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
pca = PCA(20, random_state=0).fit(Xtr); Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
gm = GM(10, reg_covar=1e-3, n_init=3, random_state=0).fit(Ztr)
means = pca.inverse_transform(gm.means_)
rng = np.random.default_rng(0); samp = np.array([pca.inverse_transform(rng.multivariate_normal(gm.means_[k], gm.covariances_[k])[None, :])[0] for k in range(10)])
s = gm.score_samples(Zte); low = np.argsort(s)[:10]
fig, axs = plt.subplots(3, 10, figsize=(5.4, 1.85), dpi=240)
rows = [means, samp, Xte[low]]
for r_ in range(3):
    for c_ in range(10):
        a = axs[r_, c_]; a.imshow(np.clip(rows[r_][c_], 0, 16).reshape(8, 8), cmap='gray_r', vmin=0, vmax=16); a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.4); sp_.set_color(GRID)
        if r_ == 2: a.set_xlabel(f'{int(yte[low[c_]])}', fontsize=8.5, labelpad=1)
        if r_ == 1: a.set_xlabel('')
for r_, t in enumerate(['means', 'samples', 'least likely\n(true digit)']): axs[r_, 0].set_ylabel(t, fontsize=8.5, rotation=90, labelpad=2)
fig.subplots_adjust(left=0.1, right=0.995, top=0.995, bottom=0.08, wspace=0.08, hspace=0.12); sv(fig, 'f4132.png', 470)
NUM['digits_low_scores'] = s[low].round(1).tolist(); NUM['digits_low_digits'] = [int(v) for v in yte[low]]
dump('fig_numbers5.json', NUM); dump('fig_textsize5.json', CHK)

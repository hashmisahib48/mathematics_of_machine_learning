# Figures 47.9-1, 47.9-2, 47.10-1, 47.10-2. Run: python3 fig/figs2.py
from common import *
from matplotlib.patches import Rectangle
from sklearn.datasets import load_iris, load_digits
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler

sdX = float(X.std()); h_scott = float(len(X) ** -0.2 * sdX)
# ---- 47.9-1  the knob turned
fig = plt.figure(figsize=(4.9, 2.95), dpi=240); gsp = fig.add_gridspec(2, 4, height_ratios=[1, 1.35])
axs = [fig.add_subplot(gsp[0, 0])]; axs += [fig.add_subplot(gsp[0, j], sharey=axs[0]) for j in range(1, 4)]
hs = [(0.8, 'h = 0.8'), (2.5, 'h = 2.5 (CV)'), (h_scott, 'h = %.1f (Scott)' % h_scott), (10.0, 'h = 10')]
for ax, (h, t) in zip(axs, hs):
    ax.plot(XGRID, truth(XGRID), color=GREY, lw=1.0, ls='--')
    ax.plot(XGRID, kde_curve(X, h), color=GREEN, lw=1.6)
    ax.plot(X[:, 0], np.full(len(X), -0.004), '|', color=BLUE, ms=3.5, mew=0.4, clip_on=False)
    ax.set_title(t, fontsize=8.5, color=PUR); ax.set_xlim(35, 105); ax.set_xticks([40, 70, 100]); ax.set_ylim(0, 0.062)
axs[0].set_yticks([0, 0.02, 0.04]); axs[0].set_ylabel('density (per min)', fontsize=8.5)
for ax in axs[1:]: plt.setp(ax.get_yticklabels(), visible=False)
top_axes = axs

# ---- 47.9-2  cross-validation and bias / variance
cv = KFold(10, shuffle=True, random_state=0)
cvres = {}
for kernel, lo in [('gaussian', 1.0), ('epanechnikov', 3.0)]:
    hh = np.arange(lo, 10.01, 0.25)
    gs = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': hh}, cv=cv).fit(X)
    cvres[kernel] = (hh, gs.cv_results_['mean_test_score'] / 20)
# Monte Carlo: 200 data sets of 200 waits each
g = np.arange(20, 120.1, 0.5); hgrid = np.arange(0.5, 10.01, 0.5); R = 200
est = np.zeros((len(hgrid), R, len(g)))
for r in range(R):
    d = waits(200, 5000 + r)[:, 0]
    D2 = (g[:, None] - d[None]) ** 2
    for j, h in enumerate(hgrid):
        est[j, r] = np.exp(-D2 / (2 * h * h)).sum(1) / (200 * h * np.sqrt(2 * np.pi))
bias2 = np.trapezoid((est.mean(1) - truth(g)) ** 2, g, axis=1)
var = np.trapezoid(est.var(1), g, axis=1)
mise = bias2 + var
NUM['mc'] = {'h': hgrid.tolist(), 'bias2': bias2.tolist(), 'var': var.tolist(), 'mise': mise.tolist(),
             'h_best': float(hgrid[np.argmin(mise)]), 'mise_best': float(mise.min())}
NUM['cv'] = {k: {'h_best': float(v[0][np.nanargmax(np.where(np.isfinite(v[1]), v[1], -np.inf))]),
                 'best': float(np.nanmax(np.where(np.isfinite(v[1]), v[1], -np.inf)))} for k, v in cvres.items()}
axs = [fig.add_subplot(gsp[1, 0:2]), fig.add_subplot(gsp[1, 2:4])]
ax = axs[0]
for k, c, ls in [('gaussian', GREEN, '-'), ('epanechnikov', BLUE, '--')]:
    hh, sc = cvres[k]; ok = np.isfinite(sc); ax.plot(hh[ok], sc[ok], color=c, ls=ls, lw=1.6, label=k)
    ib = np.argmax(np.where(ok, sc, -np.inf)); ax.plot(hh[ib], sc[ib], 'o', color=c, ms=5)
ax.axvline(h_scott, color=PUR, lw=1.0, ls=':'); ax.text(h_scott + 0.15, -4.37, 'Scott', color=PUR, fontsize=8.5, va='bottom', ha='left')
ax.set_ylim(-4.4, -3.7); ax.set_xlim(0.8, 10.2); ax.set_xlabel('bandwidth $h$ (min)'); ax.set_ylabel('10-fold CV, mean $\\ln\\hat f$')
ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.36), fontsize=8, handlelength=1.8)
ax = axs[1]
ax.plot(hgrid, bias2 * 1e4, color=RED, lw=1.5, label='bias$^2$'); ax.plot(hgrid, var * 1e4, color=OR, lw=1.5, ls='--', label='variance')
ax.plot(hgrid, mise * 1e4, color=NAVY, lw=1.8, label='sum'); ib = np.argmin(mise); ax.plot(hgrid[ib], mise[ib] * 1e4, 'o', color=NAVY, ms=5)
ax.set_xlabel('bandwidth $h$ (min)'); ax.set_ylabel('integrated, $\\times 10^{-4}$'); ax.set_ylim(0, 40); ax.set_xlim(0, 10.2)
ax.legend(loc='upper left', fontsize=8, handlelength=1.8, ncol=1, bbox_to_anchor=(0.12, 1.02))
fig.tight_layout(h_pad=0.8, w_pad=0.6); save(fig, 'f4791.png')

# ---- 47.10-1  irises: three densities
Z = StandardScaler().fit_transform(load_iris().data[:, 2:4]); sp = load_iris().target
gx, gy = np.meshgrid(np.linspace(-1.9, 1.9, 160), np.linspace(-1.8, 1.8, 160)); G = np.c_[gx.ravel(), gy.ravel()]
gm = GaussianMixture(3, n_init=3, reg_covar=1e-3, random_state=0).fit(Z)
panels = [('KDE, $h$ = 0.11 (CV)', np.exp(KernelDensity(bandwidth=0.11).fit(Z).score_samples(G))),
          ('KDE, $h$ = 0.43 (Scott)', np.exp(KernelDensity(bandwidth=150 ** (-1 / 6)).fit(Z).score_samples(G))),
          ('mixture, $K$ = 3', np.exp(gm.score_samples(G)))]
fig, axs = plt.subplots(1, 3, figsize=(5.2, 1.65), dpi=240, sharex=True, sharey=True)
mk = ['o', 's', '^']
for ax, (t, f) in zip(axs, panels):
    ax.contourf(gx, gy, f.reshape(gx.shape), levels=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], colors=[plt.cm.Greens(v) for v in np.linspace(0.18, 0.85, 9)], extend='max')
    ax.contour(gx, gy, f.reshape(gx.shape), levels=[0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4], colors=GREEN, linewidths=0.4)
    for s_, m in enumerate(mk): ax.scatter(Z[sp == s_, 0], Z[sp == s_, 1], s=7, marker=m, color=BLUE, lw=0, alpha=0.9)
    ax.set_title(t, fontsize=8.5, color=NAVY); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_xlim(-1.9, 1.9); ax.set_ylim(-1.8, 1.8)
    NUM.setdefault('iris_peak', {})[t] = float(f.max())
axs[0].set_ylabel('petal width (scaled)', fontsize=8.5)
for ax in axs: ax.set_xlabel('petal length (scaled)', fontsize=8.5)
fig.tight_layout(w_pad=0.4); save(fig, 'f47101.png')

# ---- 47.10-2  digits
Xd, yd = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
pca = PCA(20, random_state=0).fit(Xtr); Ztr = pca.transform(Xtr)
rows = [('test digits', Xte[:10])]
for h in [3.0]:
    Zn = KernelDensity(bandwidth=h).fit(Ztr).sample(10, random_state=3)
    rows.append(('KDE, $h$ = %g' % h, pca.inverse_transform(Zn)))
gmx = GaussianMixture(10, reg_covar=1e-3, n_init=3, random_state=0).fit(Ztr)
rows.append(('mixture, $K$ = 10', pca.inverse_transform(gmx.sample(10)[0])))
fig, axs = plt.subplots(len(rows), 10, figsize=(4.9, 1.6), dpi=240)
for i, (lab, im) in enumerate(rows):
    for j in range(10):
        a = axs[i, j]; a.imshow(np.clip(im[j], 0, 16).reshape(8, 8), cmap='gray_r', vmin=0, vmax=16); a.axis('off')
    axs[i, 0].text(-0.25, 0.5, lab, transform=axs[i, 0].transAxes, ha='right', va='center', fontsize=8.5, color=NAVY)
fig.subplots_adjust(left=0.2, right=0.995, top=0.995, bottom=0.005, wspace=0.05, hspace=0.05)
save(fig, 'f47102.png')
dump('fig_numbers2'); print(json.dumps({k: v for k, v in NUM.items() if k != 'mc'}), NUM['mc']['h_best'], NUM['mc']['mise_best'])

# Figures of Chapter 35. Run: python3 fig/figs.py   (numbers -> verify/fig_numbers.json)
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, Polygon
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectra import make_spectra, FAT, PROT, WATER, CH
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
NUM = {}; CHK = {}
CM = LinearSegmentedColormap.from_list('by', [BLUE, '#DDDDDD', OR])

def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name)
    from PIL import Image
    w = Image.open(name).size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1),
                 'min_font_printed': round(fs * printed_px / 96 / w, 1)}

which = sys.argv[1:] or ['f350', 'f351', 'f355', 'f356', 'f359', 'f3511']

# ---------------------------------------------------------------- f350 : PCA direction against PLS direction
if 'f350' in which:
    rng = np.random.default_rng(3); n = 90
    z = rng.standard_normal((n, 2))
    X = np.c_[2.5 * z[:, 0], 1.0 * z[:, 1] + 0.2 * z[:, 0]]
    X -= X.mean(0)
    y = 1.0 * X[:, 1] + 0.3 * rng.standard_normal(n); y -= y.mean()
    lam, V = np.linalg.eigh(X.T @ X); pca = V[:, -1]; pca *= np.sign(pca @ X.T @ y)
    g = X.T @ y; pls = g / np.linalg.norm(g)
    tp, tl = X @ pca, X @ pls
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    NUM['f350'] = {'pca': pca.round(3).tolist(), 'pls': pls.round(3).tolist(), 'var_pca': float(tp.var(ddof=1)), 'var_pls': float(tl.var(ddof=1)),
                   'r_pca': r(tp, y), 'r_pls': r(tl, y), 'angle': float(np.degrees(np.arccos(abs(pca @ pls))))}
    fig, axs = plt.subplots(1, 3, figsize=(5.6, 2.2), dpi=240, gridspec_kw={'width_ratios': [2.0, 1, 1]})
    ax = axs[0]
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CM, s=16, edgecolor='#666666', lw=0.3, zorder=4, vmin=-np.abs(y).max(), vmax=np.abs(y).max())
    L = 5.2
    arr(ax, (0, 0), tuple(L * pca), c=GREY, lw=1.3, ms=8, ls='--'); arr(ax, (0, 0), tuple(L * pls), c=GREEN, lw=1.6, ms=8)
    ax.text(L * pca[0] + 0.3, L * pca[1] + 0.2, 'PCA', color=GREY, fontsize=9, ha='left', weight='bold')
    ax.text(L * pls[0] + 0.2, L * pls[1] - 0.2, 'PLS', color=GREEN, fontsize=9, ha='left', weight='bold')
    ax.set_aspect('equal'); ax.set_xlim(-8, 8.2); ax.set_ylim(-3.2, 6.3); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$')
    ax.set_title('direction of the first component', fontsize=9, color=NAVY)
    for ax_, t, nm, c_ in ((axs[1], tp, 'PCA score', GREY), (axs[2], tl, 'PLS score', GREEN)):
        ax_.scatter(t, y, c=y, cmap=CM, s=14, edgecolor='#666666', lw=0.3, zorder=4, vmin=-np.abs(y).max(), vmax=np.abs(y).max())
        b = np.polyfit(t, y, 1); xx = np.array([t.min(), t.max()]); ax_.plot(xx, np.polyval(b, xx), color=c_, lw=1.4, zorder=3)
        ax_.set_xlabel(nm); ax_.set_title(f'$r$ = {r(t, y):.2f}', fontsize=9.5, color=c_)
        ax_.grid(color=GRID, lw=0.5); ax_.set_axisbelow(True)
    axs[1].set_ylabel('target $y$'); axs[2].set_yticklabels([])
    axs[2].set_ylim(axs[1].get_ylim()); fig.tight_layout(w_pad=0.7); sv(fig, 'f350.png', 470)

# ---------------------------------------------------------------- f351 : the spectra
if 'f351' in which:
    Xs, ys = make_spectra(60, 0)
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.2), dpi=240, gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axs[0]; o = np.argsort(ys)
    for i in o:
        ax.plot(CH, Xs[i], color=CM((ys[i] - ys.min()) / (ys.max() - ys.min())), lw=0.7, alpha=0.9)
    ax.set_xlabel('channel'); ax.set_ylabel('absorbance'); ax.set_title('60 spectra, coloured by fat content', fontsize=9, color=NAVY)
    ax.set_xlim(0, 99)
    ax = axs[1]
    for s_, nm, c_, ls in ((FAT, 'fat', OR, '-'), (PROT, 'protein', BLUE, '--'), (WATER, 'water', GREY, ':')):
        ax.plot(CH, s_ * 100, color=c_, lw=1.5, ls=ls, label=nm)
    ax.set_xlabel('channel'); ax.set_ylabel('absorbance per 100 %'); ax.set_title('pure components', fontsize=9, color=NAVY)
    ax.legend(loc='upper right', fontsize=8.5, handlelength=1.8); ax.set_xlim(0, 99)
    fig.tight_layout(w_pad=0.9); sv(fig, 'f351.png', 470)
    NUM['f351'] = {'y_min': float(ys.min()), 'y_max': float(ys.max()), 'x_min': float(Xs.min()), 'x_max': float(Xs.max())}

# ---------------------------------------------------------------- f355 : the four samples and the deflation
if 'f355' in which:
    X = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]]); y = np.array([0., -1, 2, -1])
    w1 = np.array([0.6, 0.8]); t1 = X @ w1; p1 = X.T @ t1 / (t1 @ t1); X1 = X - np.outer(t1, p1); y1 = y - 0.5 * t1
    lam, V = np.linalg.eigh(X.T @ X); pca = V[:, -1] * np.sign(V[0, -1])
    NUM['f355'] = {'X1': X1.round(4).tolist(), 'y1': y1.round(4).tolist(), 'pca': pca.round(4).tolist()}
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.3), dpi=240, sharex=True, sharey=True)
    for ax, M, yy, ttl in ((axs[0], X, y, 'centred samples, label = $y$'), (axs[1], X1, y1, 'after deflation, label = $y_1$')):
        ax.axhline(0, color=GRID, lw=0.7); ax.axvline(0, color=GRID, lw=0.7)
        ax.scatter(M[:, 0], M[:, 1], s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
        for (a, b), v in zip(M, yy):
            ax.text(a - 0.2, b - 0.3, f'{v:g}'.replace('-', '−'), fontsize=9, color=NAVY, ha='right', va='top')
        ax.set_title(ttl, fontsize=9, color=NAVY); ax.set_xlabel('$x_1$')
    L = 3.7
    arr(axs[0], (0, 0), tuple(L * pca), c=GREY, lw=1.3, ms=8, ls='--'); arr(axs[0], (0, 0), tuple(L * w1), c=GREEN, lw=1.6, ms=8)
    axs[0].text(L * pca[0] + 0.15, L * pca[1] + 0.2, 'PCA', color=GREY, fontsize=9, ha='left', weight='bold')
    axs[0].text(L * w1[0] + 0.2, L * w1[1] - 0.1, '$\\mathbf{w}_1$', color=GREEN, fontsize=10.5, ha='left')
    w2 = np.array([0.8, -0.6]); arr(axs[1], (0, 0), tuple(2.4 * w2), c=GREEN, lw=1.6, ms=8)
    axs[1].text(2.4 * w2[0] + 0.15, 2.4 * w2[1] - 0.1, '$\\mathbf{w}_2$', color=GREEN, fontsize=10.5, ha='left', va='top')
    axs[0].set_ylabel('$x_2$'); axs[0].set_xlim(-4.4, 4.9); axs[0].set_ylim(-2.2, 3.7)
    for ax in axs: ax.set_aspect('equal'); ax.spines['left'].set_visible(True)
    fig.tight_layout(w_pad=0.8); sv(fig, 'f355.png', 470)


# ---------------------------------------------------------------- f356 : flow chart of fit
if 'f356' in which:
    fig, ax = new(5.0, 4.55); ax.set_aspect('auto')
    FS = 9.0
    def fb(xc, yc, w, h, t, fc=LB, ec=BLUE, fs=FS): box(ax, xc - w / 2, yc - h / 2, w, h, t, fc=fc, ec=ec, fs=fs, r=0.07)
    A_ = lambda p, q, **k: arr(ax, p, q, c=NAVY, lw=0.9, ms=8, **k)
    cx, bw = 4.2, 7.9
    fb(cx, 9.9, bw, 0.9, 'Input: $\\mathbf{X}$ ($n\\times p$), $\\mathbf{Y}$ ($n\\times L$),\nnumber of components $A$', fc=LGREY, ec=GREY)
    fb(cx, 8.65, bw, 0.9, 'Centre every column; with scale=True divide it by\nits standard deviation (ddof = 1)')
    ax.add_patch(Rectangle((0.0, 3.4), 9.5, 4.6, fc='none', ec=PUR, lw=1.0, ls='--'))
    ax.text(0.2, 7.72, 'for  $k = 1, \\dots, A$', fontsize=FS, color=PUR, va='center', weight='bold'); ax.text(4.45, 7.72, 'start: $\\mathbf{u}$ = first column of $\\mathbf{Y}$', fontsize=FS, color=NAVY, va='center')
    fb(cx, 6.75, bw - 0.4, 1.05, '$\\mathbf{w}=\\mathbf{X}^{\\mathrm{T}}\\mathbf{u}/\\mathbf{u}^{\\mathrm{T}}\\mathbf{u}$, normalise;  $\\mathbf{t}=\\mathbf{X}\\mathbf{w}$\n$\\mathbf{c}=\\mathbf{Y}^{\\mathrm{T}}\\mathbf{t}/\\mathbf{t}^{\\mathrm{T}}\\mathbf{t}$;  $\\mathbf{u}=\\mathbf{Y}\\mathbf{c}/\\mathbf{c}^{\\mathrm{T}}\\mathbf{c}$', fc='#E3F4EA', ec=GREEN)
    fb(cx, 5.5, bw - 0.4, 0.7, '$\\|\\mathbf{w}-\\mathbf{w}_{\\mathrm{old}}\\|^{2}<$ tol, or $L=1$ ?', fc='#FDF0E0', ec=OR)
    fb(cx, 4.2, bw - 0.4, 0.95, 'sign rule;  $\\mathbf{t}=\\mathbf{X}\\mathbf{w}$;  $\\mathbf{p}=\\mathbf{X}^{\\mathrm{T}}\\mathbf{t}/\\mathbf{t}^{\\mathrm{T}}\\mathbf{t}$;\n$\\mathbf{q}_k=\\mathbf{Y}^{\\mathrm{T}}\\mathbf{t}/\\mathbf{t}^{\\mathrm{T}}\\mathbf{t}$;  deflate $\\mathbf{X}$ and $\\mathbf{Y}$', fc='#E3F4EA', ec=GREEN)
    fb(cx, 2.15, bw, 1.05, '$\\mathbf{R}=\\mathbf{W}(\\mathbf{P}^{\\mathrm{T}}\\mathbf{W})^{-1}$,  $\\mathbf{B}=\\mathbf{R}\\mathbf{Q}^{\\mathrm{T}}$;  undo the scaling:\ncoef_ and intercept_; return the model', fc='#F1EAF7', ec=PUR)
    A_((cx, 9.45), (cx, 9.1)); A_((cx, 8.2), (cx, 7.3)); A_((cx, 6.22), (cx, 5.85))
    A_((cx, 5.15), (cx, 4.68)); ax.text(cx + 0.15, 4.92, 'yes', fontsize=FS, color=NAVY, va='center')
    A_((cx, 3.4), (cx, 2.7))
    xr = cx + (bw - 0.4) / 2
    ax.plot([xr, xr + 0.5, xr + 0.5], [5.5, 5.5, 6.55], color=NAVY, lw=0.9); A_((xr + 0.5, 6.55), (xr, 6.55)); ax.text(xr + 0.08, 5.72, 'no', fontsize=FS, color=NAVY)
    ax.plot([xr, xr + 1.1, xr + 1.1], [4.2, 4.2, 6.95], color=PUR, lw=0.9); arr(ax, (xr + 1.1, 6.95), (xr, 6.95), c=PUR, lw=0.9, ms=8); ax.text(xr + 0.18, 4.4, '$k<A$', fontsize=FS, color=PUR)
    ax.set_xlim(0, 9.55); ax.set_ylim(1.4, 10.4); sv(fig, 'f356.png', 420)

# ---------------------------------------------------------------- f359 : coefficient gallery and the CV curve
if 'f359' in which:
    Xs, ys = make_spectra(60, 0); Xt, yt = make_spectra(1000, 1)
    cv = KFold(5, shuffle=True, random_state=0); ks = list(range(1, 16))
    cv_pls, cv_pcr, te_pls, tr_pls, te_pcr = [], [], [], [], []
    for k in ks:
        pl = PLSRegression(k); pc = make_pipeline(StandardScaler(), PCA(k), LinearRegression())
        cvs = lambda m: -cross_val_score(m, Xs, ys, cv=cv, scoring='neg_root_mean_squared_error').mean()
        cv_pls.append(cvs(pl)); cv_pcr.append(cvs(pc))
        pl.fit(Xs, ys); pc.fit(Xs, ys)
        te_pls.append(rmse(yt, pl.predict(Xt).ravel())); tr_pls.append(rmse(ys, pl.predict(Xs).ravel())); te_pcr.append(rmse(yt, pc.predict(Xt)))
    rid = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 30)))
    cv_rid = cvs(rid)
    NUM['f359'] = {'k': ks, 'cv_pls': np.round(cv_pls, 3).tolist(), 'cv_pcr': np.round(cv_pcr, 3).tolist(), 'test_pls': np.round(te_pls, 3).tolist(),
                   'train_pls': np.round(tr_pls, 3).tolist(), 'test_pcr': np.round(te_pcr, 3).tolist(), 'cv_ridge': float(cv_rid),
                   'best_pls': int(np.argmin(cv_pls)) + 1, 'best_pcr': int(np.argmin(cv_pcr)) + 1,
                   'min_pls': float(min(cv_pls)), 'min_pcr': float(min(cv_pcr))}
    # gallery of coefficient vectors
    fig, axs = plt.subplots(2, 3, figsize=(5.4, 3.55), dpi=240, sharex=True)
    ref = FAT / FAT.max()
    for ax, k in zip(axs.ravel(), [1, 2, 3, 4, 6, 15]):
        pl = PLSRegression(k).fit(Xs, ys); b = pl.coef_.ravel() / Xs.std(0, ddof=1) * 0 + pl.coef_.ravel()
        # coef_ is in the units of the raw channels (per unit absorbance)
        ax.plot(CH, ref * np.abs(b).max(), color=GREY, lw=0.9, ls=':', zorder=1)
        ax.plot(CH, b, color=GREEN, lw=1.2, zorder=3); ax.axhline(0, color=GRID, lw=0.6, zorder=0)
        ax.set_title(f'$A$ = {k}   (CV RMSE {cv_pls[k - 1]:.2f})', fontsize=9, color=NAVY)
        ax.set_xlim(0, 99)
    for ax in axs[1]: ax.set_xlabel('channel')
    for ax in axs[:, 0]: ax.set_ylabel('coef_')
    fig.tight_layout(h_pad=0.8, w_pad=0.6); sv(fig, 'f3591.png', 470)
    # validation curve
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.3), dpi=240)
    ax = axs[0]
    ax.plot(ks, cv_pls, color=GREEN, lw=1.6, marker='o', ms=3.5, label='PLS'); ax.plot(ks, cv_pcr, color=OR, lw=1.4, ls='--', marker='s', ms=3.5, label='PCA + OLS')
    ax.axhline(cv_rid, color=GREY, lw=1.0, ls=':'); ax.text(1, cv_rid - 0.08, 'ridge (CV RMSE %.2f)' % cv_rid, color=GREY, fontsize=9, ha='left', va='top')
    ax.set_ylim(0, 3.2); ax.set_xlabel('components $A$'); ax.set_ylabel('5-fold CV RMSE'); ax.legend(loc='upper right', fontsize=8.8)
    ax.set_title('choosing $A$ by cross-validation', fontsize=9, color=NAVY); ax.grid(color=GRID, lw=0.5)
    ax = axs[1]
    ax.plot(ks, tr_pls, color=BLUE, lw=1.4, marker='o', ms=3.5, label='training'); ax.plot(ks, te_pls, color=GREY, lw=1.6, ls='--', marker='s', ms=3.5, label='test (1000 new)')
    ax.set_ylim(0, 3.2); ax.set_xlabel('components $A$'); ax.set_ylabel('RMSE'); ax.legend(loc='upper right', fontsize=8.8)
    ax.set_title('PLS: training and test error', fontsize=9, color=NAVY); ax.grid(color=GRID, lw=0.5)
    fig.tight_layout(w_pad=1.0); sv(fig, 'f3592.png', 470)

# ---------------------------------------------------------------- f3511 : case study picture
if 'f3511' in which:
    from sklearn.cross_decomposition import CCA
    from sklearn.datasets import load_linnerud
    Xs, ys = make_spectra(60, 0); Xt, yt = make_spectra(1000, 1)
    pl = PLSRegression(6).fit(Xs, ys); ols = make_pipeline(StandardScaler(), LinearRegression()).fit(Xs, ys)
    fig, axs = plt.subplots(1, 2, figsize=(5.5, 2.45), dpi=240)
    ax = axs[0]; m = slice(0, 250)
    ax.scatter(yt[m], ols.predict(Xt[m]), s=9, facecolor='none', edgecolor=OR, lw=0.6, marker='s', label=f'OLS  (RMSE {rmse(yt, ols.predict(Xt)):.2f})', zorder=3)
    ax.scatter(yt[m], pl.predict(Xt[m]).ravel(), s=9, color=GREEN, lw=0, alpha=0.8, label=f'PLS, $A$ = 6  ({rmse(yt, pl.predict(Xt).ravel()):.2f})', zorder=4)
    lo, hi = yt.min() - 1, yt.max() + 1; ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls='--', zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel('measured fat (%)'); ax.set_ylabel('predicted fat (%)')
    ax.legend(loc='upper left', fontsize=8.5, handletextpad=0.3, borderaxespad=0.2); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.set_title('spectra: 250 of 1000 new samples', fontsize=9, color=NAVY)
    d = load_linnerud(); cca = CCA(1).fit(d.data, d.target); a, b = cca.transform(d.data, d.target); a, b = a[:, 0], b[:, 0]
    ax = axs[1]; ax.scatter(a, b, s=24, color=BLUE, edgecolor='white', lw=0.7, zorder=4)
    c = np.polyfit(a, b, 1); xx = np.array([a.min() - 0.2, a.max() + 0.2]); ax.plot(xx, np.polyval(c, xx), color=GREEN, lw=1.5, zorder=3)
    rr = float(np.corrcoef(a, b)[0, 1]); ax.text(0.04, 0.95, f'$r$ = {rr:.2f}', transform=ax.transAxes, fontsize=9.5, color=GREEN, va='top')
    ax.set_xlabel(r'exercise score  $\mathbf{X}\mathbf{w}$'); ax.set_ylabel(r'body score  $\mathbf{Y}\mathbf{c}$')
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_title('Linnerud: first CCA pair, 20 men', fontsize=9, color=NAVY)
    NUM['f3511'] = {'r_cca': rr, 'rmse_pls_test': float(rmse(yt, pl.predict(Xt).ravel())), 'rmse_ols_test': float(rmse(yt, ols.predict(Xt)))}
    fig.tight_layout(w_pad=1.0); sv(fig, 'f3511.png', 470)

json.dump({'numbers': NUM}, open('verify/fig_numbers_%s.json' % ('_'.join(which)), 'w'), indent=1)
json.dump(CHK, open('verify/fig_textsize_%s.json' % ('_'.join(which)), 'w'), indent=1)
print('figs ok', which); print(json.dumps(CHK, indent=0)[:600])

# Problem figures and answers of Chapter 35. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib'); sys.path.insert(0, HERE + '/../code'); os.chdir(HERE + '/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Rectangle
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from spectra import make_spectra
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; CHK = {}

def psv(fig, name, wpx=135):
    from PIL import Image
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); iw, ih = Image.open(name).size
    printed = min(round(wpx * 1.55), round(235 * iw / ih)); w_in = iw / 240
    CHK[name] = {'fig_in': round(w_in, 2), 'printed_px': printed, 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed / 96 / w_in, 1)}

def small(w=2.5, h=1.8): return plt.subplots(figsize=(w, h), dpi=240)
def grid(ax): ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)

# ---------------------------------------------------------------- 35A.1 first component by hand
X = np.array([[-3., -1], [-1, -2], [1, 2], [3, 1]]); y = np.array([-1., 0, 1, 0])
g = X.T @ y; w = g / np.linalg.norm(g); t = X @ w; tt = t @ t; p = X.T @ t / tt; q = y @ t / tt
X1 = X - np.outer(t, p); y1 = y - q * t; g2 = X1.T @ y1; w2 = g2 / np.linalg.norm(g2); w2 = w2 * np.sign(w2[np.argmax(abs(w2))])
B1 = w * q; ols = np.linalg.solve(X.T @ X, X.T @ y)
A['35A.1'] = {'Xty': g.tolist(), 'w': w.round(4).tolist(), 't': t.tolist(), 'tt': float(tt), 'p': p.round(4).tolist(), 'q': round(float(q), 4), 'B1': B1.round(4).tolist(),
              'y1': y1.round(4).tolist(), 'w2': w2.round(4).tolist(), 'ols': ols.round(4).tolist(), 'sse1': round(float(((y - X @ B1) ** 2).sum()), 4), 'sse_ols': round(float(((y - X @ ols) ** 2).sum()), 4),
              'pls2_sk': PLSRegression(2, scale=False).fit(X, y).coef_.round(4).ravel().tolist()}
fig, ax = small(2.5, 1.2)
ax.axhline(0, color=GRID, lw=0.7); ax.axvline(0, color=GRID, lw=0.7)
ax.scatter(X[:, 0], X[:, 1], s=34, color=BLUE, edgecolor='white', lw=0.8, zorder=5)
for (a, b), v in zip(X, y): ax.text(a + 0.25, b + 0.15, f'{v:g}'.replace('-', '−'), fontsize=9, color=NAVY, ha='left', va='bottom')
ax.set_xlim(-4.2, 4.6); ax.set_ylim(-2.9, 2.9); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal')
ax.text(0.97, 0.05, 'label = $y$', transform=ax.transAxes, fontsize=9, color=GREY, ha='right'); psv(fig, 'p_35A1.png', 135)

# ---------------------------------------------------------------- 35A.2 covariance of three directions (chapter data)
Xc = np.array([[-3., 1], [-2, -1], [2, 1], [3, -1]]); yc = np.array([0., -1, 2, -1])
vals = {nm: float((Xc @ np.array(d)) @ yc) for nm, d in (('x1', (1, 0)), ('x2', (0, 1)), ('pls', (0.6, 0.8)))}
th = np.degrees(np.arctan2(4, 3))
A['35A.2'] = {'t_y': vals, 'theta_deg': round(float(th), 2), 'max': 5.0}
fig, ax = small(2.5, 1.0); ang = np.linspace(0, 180, 361)
ax.plot(ang, 3 * np.cos(np.radians(ang)) + 4 * np.sin(np.radians(ang)), color=PUR, lw=1.5)
ax.axhline(0, color=GRID, lw=0.7)
for a_, v, c_, lab in ((0, 3, BLUE, '$w_1$=1'), (90, 4, OR, '$w_2$=1'), (th, 5, GREEN, 'PLS')):
    ax.plot([a_], [v], 'o', color=c_, ms=6, zorder=5)
ax.text(th + 6, 5.9, 'PLS', color=GREEN, fontsize=9, va='center'); ax.set_xlim(-8, 188); ax.set_ylim(-5.8, 6.8)
ax.set_xlabel('direction angle (degrees)'); ax.set_ylabel('$\\mathbf{t}^{\\mathrm{T}}\\mathbf{y}$'); ax.set_xticks([0, 90, 180]); grid(ax); psv(fig, 'p_35A2.png', 135)

# ---------------------------------------------------------------- 35A.3 unscaling
Bs = np.array([0.5, 0.25]); sx = np.array([2., 4.]); sy = 8.0; xm = np.array([10., 5.]); ym = 20.0
coef = sy * Bs / sx; xnew = np.array([12., 9.]); pred = ym + (xnew - xm) @ coef
A['35A.3'] = {'coef': coef.tolist(), 'intercept': ym, 'pred': float(pred)}
fig, ax = small(2.5, 1.15)
ax.bar([0 - 0.18, 1 - 0.18], Bs, width=0.34, color=BLUE, label='$B_j$ (scaled)'); ax.bar([0 + 0.18, 1 + 0.18], coef, width=0.34, color=GREEN, label='coef_')
ax.set_xticks([0, 1]); ax.set_xticklabels(['$x_1$', '$x_2$']); ax.legend(loc='upper right', fontsize=9); ax.set_ylim(0, 3.2); ax.set_ylabel('weight'); grid(ax); psv(fig, 'p_35A3.png', 135)

# ---------------------------------------------------------------- 35B.1 components needed, ten data sets
cv = KFold(5, shuffle=True, random_state=0); ks = list(range(1, 13)); need_pls, need_pcr = [], []
for seed in range(10):
    Xs, ys = make_spectra(60, 100 + seed)
    cp, cc = [], []
    for k in ks:
        sc_ = lambda m: -cross_val_score(m, Xs, ys, cv=cv, scoring='neg_root_mean_squared_error').mean()
        cp.append(sc_(PLSRegression(k))); cc.append(sc_(make_pipeline(StandardScaler(), PCA(k), LinearRegression())))
    need_pls.append(int(next(k for k, v in zip(ks, cp) if v <= 1.5))); need_pcr.append(int(next(k for k, v in zip(ks, cc) if v <= 1.5)))
A['35B.1'] = {'threshold': 1.5, 'need_pls': need_pls, 'need_pcr': need_pcr, 'median_pls': float(np.median(need_pls)), 'median_pcr': float(np.median(need_pcr))}
fig, ax = small(2.5, 1.15)
for i, (a_, b_) in enumerate(zip(need_pls, need_pcr)):
    ax.plot([a_, b_], [i, i], color=GRID, lw=1.0, zorder=1)
ax.scatter(need_pls, range(10), s=22, color=GREEN, zorder=3, label='PLS'); ax.scatter(need_pcr, range(10), s=22, color=OR, marker='s', zorder=3, label='PCA + OLS')
ax.set_xlabel('components needed'); ax.set_ylabel('data set'); ax.legend(loc='center left', fontsize=8.8, borderaxespad=0.1); ax.set_yticks([0, 5, 9]); grid(ax); psv(fig, 'p_35B1.png', 135)

# ---------------------------------------------------------------- 35B.2 scale sensitivity on Linnerud
d = load_linnerud(); XL, YL = d.data.copy(), d.target
XL2 = XL.copy(); XL2[:, 1] *= 100
def wv(Xm, sc): return np.abs(PLSRegression(1, scale=sc).fit(Xm, YL).x_weights_[:, 0])
W_ = {'raw, scale=False': wv(XL, False), 'Situps x 100, scale=False': wv(XL2, False), 'Situps x 100, scale=True': wv(XL2, True), 'raw, scale=True': wv(XL, True)}
A['35B.2'] = {k: v.round(4).tolist() for k, v in W_.items()}
fig, ax = small(2.5, 1.1)
for i, (lab, c_, nm) in enumerate((('raw, scale=False', GREY, 'raw'), ('Situps x 100, scale=False', OR, '×100'), ('Situps x 100, scale=True', GREEN, '×100, scaled'))):
    ax.bar(np.arange(3) + (i - 1) * 0.26, W_[lab], width=0.24, color=c_, label=nm)
ax.set_xticks(range(3)); ax.set_xticklabels(['Chins', 'Situps', 'Jumps'], fontsize=9); ax.set_ylabel('$|w_1|$'); ax.set_ylim(0, 2.3)
ax.legend(loc='upper left', fontsize=8.6, handlelength=1.0, borderaxespad=0.1, labelspacing=0.25); grid(ax); psv(fig, 'p_35B2.png', 135)

# ---------------------------------------------------------------- 35B.3 PLS = least squares on the scores
Xs, ys = make_spectra(60, 0); pl = PLSRegression(3).fit(Xs, ys); T = pl.x_scores_
lr = LinearRegression().fit(T, ys)
A['35B.3'] = {'max_pred_diff': float(abs(lr.predict(T) - pl.predict(Xs).ravel()).max()), 'coef_on_scores': lr.coef_.round(5).tolist(), 'y_loadings_times_sd': (pl.y_loadings_.ravel() * ys.std(ddof=1)).round(5).tolist(),
              'score_corr_12': float(np.corrcoef(T[:, 0], T[:, 1])[0, 1]), 'score_norms': np.linalg.norm(T, axis=0).round(3).tolist(), 'intercept_diff': float(abs(lr.intercept_ - pl.intercept_[0]))}
fig, ax = small(2.5, 1.15)
ax.scatter(T[:, 0], T[:, 1], c=ys, cmap='coolwarm', s=14, edgecolor='#666666', lw=0.3); ax.set_xlabel('score $t_1$'); ax.set_ylabel('score $t_2$'); grid(ax); psv(fig, 'p_35B3.png', 135)

# ---------------------------------------------------------------- 35C.1 nuisance direction with huge variance
rng = np.random.default_rng(7); n = 70; z = rng.standard_normal((n, 2))
Xn = np.c_[40 * z[:, 0], 1.0 * z[:, 1]]; yn = 1.0 * Xn[:, 1] + 0.3 * rng.standard_normal(n); Xn -= Xn.mean(0); yn -= yn.mean()
pc1 = PCA(1).fit(Xn).components_[0]; gn = Xn.T @ yn; pls1 = gn / np.linalg.norm(gn)
Zn = (Xn - Xn.mean(0)) / Xn.std(0, ddof=1); gz = Zn.T @ yn; plsz = gz / np.linalg.norm(gz); pcz = PCA(1).fit(Zn).components_[0]
A['35C.1'] = {'r_pls_scaled': float(abs(np.corrcoef(Zn @ plsz, yn)[0, 1])), 'w_pls_scaled': np.abs(plsz).round(4).tolist(), 'r_pca_scaled': float(abs(np.corrcoef(Zn @ pcz, yn)[0, 1])), 'pca': np.abs(pc1).round(4).tolist(), 'pls': np.abs(pls1).round(4).tolist(), 'r_pca': float(abs(np.corrcoef(Xn @ pc1, yn)[0, 1])), 'r_pls': float(abs(np.corrcoef(Xn @ pls1, yn)[0, 1])),
              'sd': Xn.std(0, ddof=1).round(2).tolist()}
fig, ax = small(2.5, 1.15)
ax.scatter(Xn[:, 0], Xn[:, 1], c=yn, cmap='coolwarm', s=14, edgecolor='#666666', lw=0.3); ax.set_xlabel('$x_1$  (sd 40)'); ax.set_ylabel('$x_2$  (sd 1)'); grid(ax); ax.set_ylim(-3.2, 3.2); psv(fig, 'p_35C1.png', 135)

# ---------------------------------------------------------------- 35C.2 leakage: components chosen outside the loop
Xs, ys = make_spectra(60, 0); cv = KFold(5, shuffle=True, random_state=0)
Tall = PLSRegression(10).fit(Xs, ys).x_scores_
leak = rmse(ys, cross_val_predict(LinearRegression(), Tall, ys, cv=cv))
honest = rmse(ys, cross_val_predict(PLSRegression(10), Xs, ys, cv=cv).ravel())
rng = np.random.default_rng(0); Xr = rng.standard_normal((60, 100)); yr = rng.standard_normal(60)
leak_noise = rmse(yr, cross_val_predict(LinearRegression(), PLSRegression(10).fit(Xr, yr).x_scores_, yr, cv=cv))
honest_noise = rmse(yr, cross_val_predict(PLSRegression(10), Xr, yr, cv=cv).ravel())
A['35C.2'] = {'leak_spectra': float(leak), 'honest_spectra': float(honest), 'leak_noise': float(leak_noise), 'honest_noise': float(honest_noise), 'sd_noise': float(yr.std(ddof=1))}
fig, ax = small(2.5, 1.0)
ax.bar([0, 1], [leak_noise, honest_noise], color=[OR, GREEN], width=0.6); ax.axhline(yr.std(ddof=1), color=GREY, lw=1.0, ls='--'); ax.text(-0.45, yr.std(ddof=1) + 0.05, 'sd of $y$', color=GREY, fontsize=9, ha='left', va='bottom')
ax.set_xticks([0, 1]); ax.set_xticklabels(['inside-out', 'honest'], fontsize=9); ax.set_ylabel('CV RMSE'); ax.set_ylim(0, 1.65); ax.set_xlim(-0.5, 1.5); grid(ax); psv(fig, 'p_35C2.png', 135)

json.dump(A, open('verify/problem_answers.json', 'w'), indent=1); json.dump(CHK, open('verify/probfig_textsize.json', 'w'), indent=1)
print(json.dumps(A, indent=0)); print({k: v['min_font_printed'] for k, v in CHK.items()})

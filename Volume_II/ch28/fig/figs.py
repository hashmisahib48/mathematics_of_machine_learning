# All figures of Chapter 28. Run: python3 fig/figs.py   (PNGs into ch28/, numbers into verify/fig_numbers.json)
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
os.environ['OMP_NUM_THREADS'] = '1'
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons, make_classification, make_friedman1
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5})
NUM = {}; CHK = {}
ONLY = set(sys.argv[1:])
def want(k): return (not ONLY) or (k in ONLY)
def sv(fig, name, printed_px):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig, name); from PIL import Image
    with Image.open(name) as im: w = im.size[0] / 240
    CHK[name] = {'fig_in': round(w, 2), 'printed_in': round(printed_px / 96, 2), 'min_font_design': round(fs, 1), 'min_font_printed': round(fs * printed_px / 96 / w, 1)}
def grid(ax): ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
LO = '#FDEBD0'

# ---------------------------------------------------------------- f280  opener: bins, histogram, scan
if want('f280'):
    rng = np.random.default_rng(4); n = 240
    x = np.sort(rng.lognormal(0.0, 0.45, n)); y = 1.6 / (1 + np.exp(-6 * (np.log(x) - 0.25))) + rng.normal(0, 0.45, n)
    F0 = y.mean(); g = F0 - y; B = 8
    thr = np.percentile(x, np.linspace(0, 100, B + 1)[1:-1], method='averaged_inverted_cdf')
    b = np.searchsorted(thr, x); G = np.bincount(b, g, B); H = np.bincount(b, minlength=B).astype(float)
    GL = np.cumsum(G)[:-1]; HL = np.cumsum(H)[:-1]; Gt, Ht = G.sum(), H.sum()
    gain = GL**2 / HL + (Gt - GL)**2 / (Ht - HL) - Gt**2 / Ht; k = int(np.argmax(gain))
    edges = np.r_[x.min() - 0.03, thr, x.max() + 0.03]
    fig, axs = plt.subplots(3, 1, figsize=(5.0, 3.9), sharex=True, gridspec_kw={'height_ratios': [1.15, 1, 1]})
    ax = axs[0]
    ax.scatter(x, g, s=9, color=RED, alpha=0.75, lw=0, zorder=3)
    ax.axhline(0, color=GREY, lw=0.6)
    for t in thr: ax.axvline(t, color=PUR, lw=0.9, ls='--', zorder=1)
    ax.set_ylabel('gradient $g_i$'); ax.set_title('1  Bin the feature at its quantiles (purple), 240 samples, 8 bins', fontsize=9, color=NAVY, loc='left'); grid(ax)
    ax = axs[1]
    for i in range(B):
        ax.bar(edges[i], G[i], width=edges[i + 1] - edges[i], align='edge', color=RED if G[i] > 0 else PINK, ec=RED, lw=0.7, hatch=None if G[i] > 0 else '///')
        ax.text((edges[i] + edges[i + 1]) / 2, G[i] + (1.5 if G[i] > 0 else -1.5), f'{G[i]:.0f}'.replace('-', '\u2212'), ha='center', va='bottom' if G[i] > 0 else 'top', fontsize=8.5, color=NAVY)
    for t in thr: ax.axvline(t, color=PUR, lw=0.9, ls='--', zorder=1)
    ax.axhline(0, color=GREY, lw=0.6); ax.set_ylabel('bin sum $G_b$'); ax.set_ylim(G.min() - 9, G.max() + 9)
    ax.set_title('2  One pass over the samples fills the histogram (30 samples per bin)', fontsize=9, color=NAVY, loc='left'); grid(ax)
    ax = axs[2]
    ax.plot(thr, gain, '-o', color=GREEN, ms=4, lw=1.3, zorder=3); ax.plot(thr[k], gain[k], '*', color=GREEN, ms=13, mec='white', zorder=4)
    for t in thr: ax.axvline(t, color=PUR, lw=0.9, ls='--', zorder=1)
    ax.axvline(thr[k], color=GREEN, lw=1.6, zorder=2)
    ax.text(thr[k] + 0.06, gain[k] * 0.22, 'best split', fontsize=8.8, color=GREEN, ha='left')
    ax.set_ylabel('gain'); ax.set_xlabel('feature value $x$'); ax.set_ylim(0, gain.max() * 1.15)
    ax.set_title('3  Scan the 7 thresholds from left to right: 7 gains, not 239', fontsize=9, color=NAVY, loc='left'); grid(ax)
    fig.tight_layout(h_pad=0.6); sv(fig, 'f280.png', 440)
    NUM['f280'] = {'n': n, 'B': B, 'counts': H.astype(int).tolist(), 'G': G.round(1).tolist(), 'best_k': k, 'gain': gain.round(1).tolist()}

# toy data of Chapter 28 (Listing 28.1)
TX = np.array([8, 15, 22, 31, 40, 47, 58, 66, 73, 85, 92, 110.]); TY = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
TTHR = np.array([26.5, 52.5, 79.0]); TB = np.searchsorted(TTHR, TX)
def toy_pts(ax, ys=(0.0, 1.0), s=46):
    for cls, col, mk in ((0, BLUE, 'o'), (1, OR, 's')):
        m = TY == cls; ax.scatter(TX[m], np.where(TY[m] == 1, ys[1], ys[0]), s=s, color=col, marker=mk, edgecolor='white', lw=0.8, zorder=5)

# ---------------------------------------------------------------- f281  the twelve visits
if want('f281'):
    fig, ax = plt.subplots(figsize=(4.0, 2.15))
    toy_pts(ax)
    for t in TTHR: ax.axvline(t, color=PUR, lw=1.0, ls='--', zorder=1)
    edges = np.r_[0, TTHR, 120]
    for i in range(4):
        ax.text((edges[i] + edges[i + 1]) / 2, 1.42, f'bin {i}', ha='center', fontsize=9, color=PUR, weight='bold')
        ax.text((edges[i] + edges[i + 1]) / 2, 1.24, f'{int((TB == i).sum())} visits', ha='center', fontsize=8.5, color=GREY)
    for t in TTHR: ax.text(t + 1.2, -0.34, f'{t:g}', ha='left', fontsize=8.5, color=PUR)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['no purchase', 'bought'], fontsize=8.8)
    ax.set_xlim(0, 120); ax.set_ylim(-0.45, 1.6); ax.set_xlabel('seconds on the product page $x$'); ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False); ax.tick_params(axis='y', length=0)
    fig.tight_layout(); sv(fig, 'f281.png', 380)

# ---------------------------------------------------------------- f282  candidate thresholds: all gaps, quantile bins, equal-width bins
if want('f282'):
    rng = np.random.default_rng(3); v = np.sort(rng.lognormal(0, 0.7, 30))
    fig, axs = plt.subplots(3, 1, figsize=(4.7, 2.9), sharex=True)
    cuts = [(v[:-1] + v[1:]) / 2, np.percentile(v, np.linspace(0, 100, 7)[1:-1], method='averaged_inverted_cdf'), np.linspace(v.min(), v.max(), 7)[1:-1]]
    ttl = ['every gap between sorted values:  29 candidate thresholds', 'quantile bins (scikit-learn), 6 bins:  5 candidates', 'equal-width bins, 6 bins:  5 candidates']
    cols = [GREY, PUR, OR]
    for ax, c, t, col in zip(axs, cuts, ttl, cols):
        ax.scatter(v, np.zeros_like(v), s=14, color=BLUE, zorder=4, lw=0)
        for u in c: ax.plot([u, u], [-0.3, 0.55], color=col, lw=1.0 if len(c) > 10 else 1.6, zorder=2)
        ax.set_ylim(-0.6, 2.3); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.text(0.0, 0.97, t, transform=ax.transAxes, fontsize=9, color=NAVY, va='top')
    axs[0].text(1.0, 0.35, '', transform=axs[0].transAxes)
    axs[2].set_xlabel('feature value (30 samples, skewed)')
    nb = [int(((v > (e0)) & (v <= e1)).sum()) for e0, e1 in zip(np.r_[-np.inf, cuts[2]], np.r_[cuts[2], np.inf])]
    nq = [int(((v > (e0)) & (v <= e1)).sum()) for e0, e1 in zip(np.r_[-np.inf, cuts[1]], np.r_[cuts[1], np.inf])]
    NUM['f282'] = {'quantile_counts': nq, 'equal_width_counts': nb}
    fig.tight_layout(h_pad=0.5); sv(fig, 'f282.png', 450)

# ---------------------------------------------------------------- f283  histogram and scan of the toy data
if want('f283'):
    p = TY.mean(); g = p - TY; h = np.full(12, p * (1 - p)); G = np.bincount(TB, g, 4); H = np.bincount(TB, h, 4)
    GL = np.cumsum(G)[:-1]; HL = np.cumsum(H)[:-1]; gain = GL**2 / HL + (G.sum() - GL)**2 / (H.sum() - HL)
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.35), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axs[0]
    for i in range(4): ax.bar(i, G[i], width=0.62, color=RED if G[i] > 0 else PINK, ec=RED, lw=0.8, hatch=None if G[i] > 0 else '///', zorder=3); ax.text(i, G[i] + (0.08 if G[i] > 0 else -0.08), f'{G[i]:.2f}'.replace('-', '\u2212'), ha='center', va='bottom' if G[i] > 0 else 'top', fontsize=8.8, color=NAVY)
    ax.axhline(0, color=GREY, lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels([f'bin {i}' for i in range(4)], fontsize=8.5); ax.set_xlabel('every bin has $H_b$ = 0.73', fontsize=8.8); ax.set_ylim(-1.25, 1.75); ax.set_ylabel('bin sum $G_b$'); ax.set_title('histogram of gradients', fontsize=9, color=NAVY); grid(ax)
    ax = axs[1]
    for i in range(3): ax.bar(i, gain[i], width=0.62, color=GREEN if i == 0 else '#BFE3CD', ec=GREEN, lw=0.8, zorder=3); ax.text(i, gain[i] + 0.06, f'{gain[i]:.3f}', ha='center', fontsize=8.8, color=NAVY)
    ax.set_xticks(range(3)); ax.set_xticklabels([f'after\nbin {i}' for i in range(3)], fontsize=8.5); ax.set_ylim(0, 3.4); ax.set_ylabel('gain'); ax.set_title('gain of each split', fontsize=9, color=NAVY); grid(ax)
    fig.tight_layout(w_pad=1.2); sv(fig, 'f283.png', 440)

# ---------------------------------------------------------------- f284  histogram subtraction
if want('f284'):
    p = TY.mean(); g = p - TY; G = np.bincount(TB, g, 4); Gl = np.bincount(TB[TB <= 0], g[TB <= 0], 4); Gr = G - Gl
    fig, axs = plt.subplots(1, 3, figsize=(4.8, 1.85), sharey=True)
    data = [(G, 'parent: 12 samples', BLUE, '(known)'), (Gl, 'smaller child: 3 samples', OR, '(one pass over 3 samples)'), (Gr, 'larger child: 9 samples', GREEN, '(no pass at all)')]
    for ax, (d, t, col, sub2) in zip(axs, data):
        for i in range(4):
            if abs(d[i]) > 1e-9: ax.bar(i, d[i], width=0.6, color=col, alpha=0.85, zorder=3); ax.text(i if d[i] > 0 else i + (0.3 if i == 1 else -0.3), d[i] + 0.1 if d[i] > 0 else d[i] - 0.1, f'{d[i]:.2f}'.replace('-', '\u2212'), ha='center' if d[i] > 0 else ('right' if i == 1 else 'left'), va='bottom' if d[i] > 0 else 'top', fontsize=8, color=NAVY, weight='normal')
        ax.axhline(0, color=GREY, lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels([f'{i}' for i in range(4)], fontsize=8.5); ax.set_ylim(-1.25, 1.75); ax.set_xlim(-0.75, 3.75); ax.set_title(t, fontsize=8.5, color=NAVY); ax.set_xlabel('bin  ' + sub2, fontsize=8.3); grid(ax)
    axs[0].set_ylabel('$G_b$'); fig.tight_layout(w_pad=2.6)
    for a0, a1, sym in ((axs[0], axs[1], '\u2212'), (axs[1], axs[2], '=')): fig.text((a0.get_position().x1 + a1.get_position().x0) / 2, 0.55, sym, fontsize=16, ha='center', va='center', color=NAVY)
    sv(fig, 'f284.png', 440)

# ---------------------------------------------------------------- f285  best-first growth
if want('f285'):
    Xm, ym = make_moons(300, noise=0.3, random_state=1)
    m = HGBC(max_iter=1, learning_rate=1.0, max_leaf_nodes=8, min_samples_leaf=10, early_stopping=False, random_state=0).fit(Xm, ym)
    nd = m._predictors[0][0].nodes; thr_ok = lambda r: r
    # split order by replaying the priority queue
    order = {}; avail = [0]; k = 1
    while avail:
        j = max(avail, key=lambda a: nd['gain'][a]); avail.remove(j); order[j] = k; k += 1
        for c in (nd['left'][j], nd['right'][j]):
            if not nd['is_leaf'][c]: avail.append(c)
    pos = {}; cnt = [0]
    def lay(j, d):
        if nd['is_leaf'][j]: pos[j] = (cnt[0], -d); cnt[0] += 1
        else:
            lay(nd['left'][j], d + 1); lay(nd['right'][j], d + 1)
            pos[j] = ((pos[nd['left'][j]][0] + pos[nd['right'][j]][0]) / 2, -d)
    lay(0, 0); depth = int(nd['depth'].max())
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 3.3), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axs[0]; ax.axis('off'); fn = ['$x_1$', '$x_2$']
    for j in range(len(nd)):
        if not nd['is_leaf'][j]:
            for c in (nd['left'][j], nd['right'][j]): ax.plot([pos[j][0], pos[c][0]], [pos[j][1] - 0.12, pos[c][1] + 0.12], color=GREY, lw=0.8, zorder=1)
    for j in range(len(nd)):
        x0, y0 = pos[j]
        if nd['is_leaf'][j]:
            v = nd['value'][j]; ax.text(x0, y0, f'{v:+.1f}'.replace('-', '\u2212'), ha='center', va='center', fontsize=8.3, color=NAVY, bbox=dict(boxstyle='round,pad=0.25', fc=LO if v > 0 else LB, ec=OR if v > 0 else BLUE, lw=0.8))
        else:
            ax.text(x0, y0, f'{fn[nd["feature_idx"][j]]} \u2264 {nd["num_threshold"][j]:.2f}'.replace('-', '\u2212'), ha='center', va='center', fontsize=8.3, color=NAVY, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=GREEN, lw=1.0), zorder=3)
            ax.text(x0 - 0.02, y0 + 0.3, f'{order[j]}', ha='center', va='center', fontsize=8.3, color='white', weight='bold', bbox=dict(boxstyle='circle,pad=0.2', fc=PUR, ec='none'), zorder=4)
    ax.set_xlim(-0.6, cnt[0] - 0.4); ax.set_ylim(-depth - 0.5, 0.6)
    ax.set_title('tree of 8 leaves; circles give the order of the splits', fontsize=8.8, color=NAVY)
    ax = axs[1]
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.8, 300)); Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.unique(np.round(nd['value'][nd['is_leaf'] == 1], 6)).tolist() + [99], colors=None, cmap='RdBu_r', alpha=0.0)
    ax.pcolormesh(xx, yy, (Z > 0).astype(float), cmap=ListedColormap([LB, LO]), shading='auto', zorder=0)
    for cls, col, mk in ((0, BLUE, 'o'), (1, OR, 's')):
        mm = ym == cls; ax.scatter(Xm[mm, 0], Xm[mm, 1], s=11, color=col, marker=mk, edgecolor='white', lw=0.4, zorder=5)
    def rects(j, lo, hi):
        if nd['is_leaf'][j]: return
        f, t = nd['feature_idx'][j], nd['num_threshold'][j]
        if f == 0: ax.plot([t, t], [lo[1], hi[1]], color=GREEN, lw=1.2, zorder=4)
        else: ax.plot([lo[0], hi[0]], [t, t], color=GREEN, lw=1.2, zorder=4)
        l2, h2 = list(lo), list(hi); h2[f] = t; rects(nd['left'][j], lo, h2)
        l3, h3 = list(lo), list(hi); l3[f] = t; rects(nd['right'][j], l3, hi)
    rects(0, [-1.6, -1.2], [2.6, 1.8]); ax.set_xlim(-1.6, 2.6); ax.set_ylim(-1.2, 1.8); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_title('the same tree in the plane', fontsize=8.8, color=NAVY)
    fig.tight_layout(w_pad=0.8); sv(fig, 'f285.png', 470)
    NUM['f285'] = {'depth': depth, 'leaves': int(nd['is_leaf'].sum()), 'order': {int(k): int(v) for k, v in order.items()}, 'gains': [round(float(nd['gain'][j]), 2) for j in sorted(order, key=order.get)]}

# ---------------------------------------------------------------- f287  gallery of max_bins
if want('f287'):
    rng = np.random.default_rng(2); n = 500; x = rng.uniform(0, 10, n); y = np.sin(x) + 0.3 * x + rng.normal(0, 0.35, n)
    xt = rng.uniform(0, 10, 2000); yt = np.sin(xt) + 0.3 * xt + rng.normal(0, 0.35, 2000); xg = np.linspace(0, 10, 600)
    fig, axs = plt.subplots(2, 2, figsize=(5.0, 3.8), sharex=True, sharey=True); NUM['f287'] = {}
    for ax, mb in zip(axs.ravel(), (3, 6, 16, 255)):
        m = HGBR(max_bins=mb, random_state=0, early_stopping=False).fit(x[:, None], y); rm = float(np.sqrt(np.mean((m.predict(xt[:, None]) - yt)**2)))
        th = m._bin_mapper.bin_thresholds_[0]; NUM['f287'][mb] = {'rmse': round(rm, 3), 'bins': int(len(th) + 1)}
        ax.scatter(x, y, s=5, color=BLUE, alpha=0.45, lw=0, zorder=2); ax.plot(xg, m.predict(xg[:, None]), color=GREEN, lw=1.6, zorder=4)
        for t in th: ax.plot([t, t], [-1.25, -0.85], color=PUR, lw=0.8, zorder=3)
        ax.set_title(f'max_bins = {mb}   (test RMSE {rm:.3f})', fontsize=8.8, color=NAVY); grid(ax); ax.set_ylim(-1.4, 4.4)
    for ax in axs[1]: ax.set_xlabel('$x$')
    for ax in axs[:, 0]: ax.set_ylabel('$y$')
    fig.tight_layout(h_pad=0.7, w_pad=0.6); sv(fig, 'f287.png', 470)

# ---------------------------------------------------------------- f288  early stopping and validation curve
if want('f288'):
    from sklearn.model_selection import validation_curve
    X, y = make_classification(12000, 20, n_informative=6, flip_y=0.1, random_state=0)
    m = HGBC(max_iter=500, random_state=0).fit(X, y); tl, vl = -m.train_score_, -m.validation_score_; best = int(np.argmin(vl))
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.6))
    ax = axs[0]; it = np.arange(len(vl)); ax.plot(it, tl, color=BLUE, lw=1.4, label='training'); ax.plot(it, vl, color=GREY, lw=1.4, ls='--', label='validation')
    ax.axvline(best, color=PUR, lw=1.0); ax.text(best - 1.5, 0.5, f'best: tree {best}', color=PUR, fontsize=8.5, ha='right'); ax.axvline(len(vl) - 1, color=RED, lw=1.0); ax.text(len(vl) + 1, 0.4, f'stop: tree {len(vl) - 1}', color=RED, fontsize=8.5, ha='left')
    ax.set_xlim(0, 118); ax.set_ylim(0.0, 0.75); ax.set_xlabel('trees'); ax.set_ylabel('log loss'); ax.legend(loc='upper left', bbox_to_anchor=(0.12, 1.0)); grid(ax); ax.set_title('early stopping', fontsize=9, color=NAVY)
    NUM['f288'] = {'best': best, 'stop': len(vl) - 1, 'val_best': round(float(vl[best]), 4), 'train_last': round(float(tl[-1]), 4), 'val_last': round(float(vl[-1]), 4)}
    Xv, yv = make_classification(3000, 20, n_informative=6, flip_y=0.1, random_state=1)
    Ls = [2, 3, 4, 6, 8, 12, 16, 31, 64]
    tr, va = validation_curve(HGBC(max_iter=100, early_stopping=False, random_state=0), Xv, yv, param_name='max_leaf_nodes', param_range=Ls, cv=5, scoring='neg_log_loss')
    tr, va = -tr, -va; bi = int(np.argmin(va.mean(1)))
    ax = axs[1]
    for d, col, ls_, lab in ((tr, BLUE, '-', 'training folds'), (va, GREY, '--', 'validation folds')):
        ax.plot(Ls, d.mean(1), ls_, color=col, lw=1.4, marker='o', ms=3, label=lab)
    ax.fill_between(Ls, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=GREY, alpha=0.2, lw=0); ax.axvline(Ls[bi], color=PUR, lw=1.0)
    ax.set_xscale('log', base=2); ax.set_xticks(Ls); ax.set_xticklabels([str(v) for v in Ls], fontsize=7.6, rotation=0); ax.minorticks_off(); ax.set_xlabel('max_leaf_nodes'); ax.set_ylabel('log loss'); ax.legend(loc='lower left'); grid(ax); ax.set_title('validation curve', fontsize=9, color=NAVY)
    NUM['f288']['curve'] = {'L': Ls, 'train': tr.mean(1).round(4).tolist(), 'val': va.mean(1).round(4).tolist(), 'best_L': Ls[bi]}
    fig.tight_layout(w_pad=1.0); sv(fig, 'f288.png', 470)

# ---------------------------------------------------------------- f289  speed
if want('f289'):
    with open('verify/speed.json') as fh: sp = json.load(fh)
    n = np.array(sp['n'])
    fig, ax = plt.subplots(figsize=(3.9, 2.6))
    for key, col, mk, ls_, lab in (('gbc', OR, 's', '-', 'GradientBoosting, classes'), ('gbr', OR, '^', '--', 'GradientBoosting, regression'), ('hgbc', GREEN, 'o', '-', 'HistGradientBoosting, classes'), ('hgbr', GREEN, 'v', '--', 'HistGradientBoosting, regression')):
        pts_ = [(a, b) for a, b in zip(n, sp[key]) if b is not None]; ax.plot(*zip(*pts_), ls_, color=col, marker=mk, ms=4, lw=1.4, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('training samples $n$'); ax.set_ylabel('fit time (s), 100 trees'); ax.legend(fontsize=7.8, loc='upper left'); grid(ax); ax.set_ylim(0.1, 300)
    fig.tight_layout(); sv(fig, 'f289.png', 380)

# ---------------------------------------------------------------- f286  flow chart of fit
if want('f286'):
    fig, ax = new(4.8, 2.9)
    bw, bh, gx = 2.75, 1.85, 0.55
    xs = [k * (bw + gx) for k in range(4)]
    y1, y2 = 5.6, 1.05
    def fb(col, yc, t, fc=LB, ec=BLUE, fs=8.2): box(ax, xs[col], yc - bh / 2, bw, bh, t, fc=fc, ec=ec, fs=fs, r=0.06)
    A = lambda a_, b_: arr(ax, a_, b_, c=NAVY, lw=0.9, ms=8)
    fb(0, y1, 'Bin the features;\nset a validation\npart aside', fc=LGREY, ec=GREY)
    fb(1, y1, 'Start with the\nbest constant\n$F_0$')
    fb(2, y1, 'Next tree:\ngradients $g_i$,\nhessians $h_i$', fc='#FDF0E0', ec=OR)
    fb(3, y1, 'Root: one pass\nfills the\nhistograms', fc='#E3F4EA', ec=GREEN)
    fb(3, y2, 'Split best leaf:\nsmall child: pass,\nlarge: subtract', fc='#E3F4EA', ec=GREEN)
    fb(2, y2, 'Leaf values\n$-\\eta\\,G/(H+\\lambda)$;\nupdate $F$', fc='#E3F4EA', ec=GREEN)
    fb(1, y2, 'Validation loss\nstill improving?', fc='#F1EAF7', ec=PUR)
    fb(0, y2, 'Keep all trees:\n$F_0$ + their sum', fc=LGREY, ec=GREY)
    for c_ in range(3): A((xs[c_] + bw, y1), (xs[c_ + 1], y1))
    A((xs[3] + bw / 2, y1 - bh / 2), (xs[3] + bw / 2, y2 + bh / 2))
    A((xs[3], y2), (xs[2] + bw, y2)); A((xs[2], y2), (xs[1] + bw, y2)); A((xs[1], y2), (xs[0] + bw, y2))
    ax.text(xs[1] - gx / 2, y2 + 0.3, 'no', fontsize=8.2, color=NAVY, ha='center')
    ax.text(xs[3] + bw + 0.12, 2.75, 'until `max_leaf_nodes`\nor no positive gain'.replace('`', ''), fontsize=7.8, color=NAVY, ha='left', va='center') if False else None
    ym = (y1 + y2) / 2
    ax.plot([xs[1] + bw / 2, xs[1] + bw / 2, xs[2] + bw / 2], [y2 + bh / 2, ym, ym], color=NAVY, lw=0.9); A((xs[2] + bw / 2, ym), (xs[2] + bw / 2, y1 - bh / 2))
    ax.text(xs[1] + bw / 2 + 0.12, ym + 0.2, 'yes: next tree', fontsize=8.2, color=NAVY, ha='left')
    ax.text(xs[3] + bw / 2 + 0.12, ym + 0.2, 'repeat per leaf', fontsize=8.2, color=NAVY, ha='left')
    ax.set_xlim(-0.1, xs[3] + bw + 0.1); ax.set_ylim(0.0, 6.7); sv(fig, 'f286.png', 450)

# ---------------------------------------------------------------- f2811  case study figures
if want('f2811'):
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.datasets import load_breast_cancer, load_diabetes
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X0, y0 = load_breast_cancer(return_X_y=True); rng = np.random.default_rng(0)
    holes = {'complete': np.zeros(X0.shape, bool), '15% at random': rng.random(X0.shape) < 0.15}
    skip = np.zeros(X0.shape, bool)
    for j in range(20, 30): skip[:, j] = rng.random(len(y0)) < np.where(y0 == 1, 0.5, 0.05)
    holes['skipped if benign'] = skip
    hgb = HGBC(min_samples_leaf=10, l2_regularization=1, random_state=0)
    lr = lambda ind: make_pipeline(SimpleImputer(strategy='median', add_indicator=ind), StandardScaler(), LogisticRegression(max_iter=1000))
    models = {'HGB, NaN as is': hgb, 'HGB, median fill': make_pipeline(SimpleImputer(strategy='median'), hgb), 'LogReg, median': lr(False), 'LogReg, median + flags': lr(True)}
    res = {}
    for name, mask in holes.items():
        X = np.where(mask, np.nan, X0); Xtr, Xte, ytr, yte = train_test_split(X, y0, test_size=0.25, random_state=0, stratify=y0)
        res[name] = [cross_val_score(m, Xtr, ytr, cv=5).mean() for m in models.values()]
    fig, ax = plt.subplots(figsize=(4.7, 2.5)); w = 0.2; cols = [GREEN, '#9BD3B1', BLUE, '#9DB9E8']; hat = [None, None, '//', '//']
    for k, (lab, c_, h_) in enumerate(zip(models, cols, hat)):
        v = [res[n][k] for n in holes]; xs = np.arange(3) + (k - 1.5) * w
        ax.bar(xs, v, width=w * 0.92, color=c_, ec=NAVY, lw=0.5, hatch=h_, label=lab, zorder=3)
    ax.set_xticks(range(3)); ax.set_xticklabels(list(holes)); ax.set_ylim(0.93, 1.0); ax.set_ylabel('5-fold accuracy'); ax.legend(fontsize=7.8, loc='upper left', ncol=2, columnspacing=1.0, handlelength=1.2); grid(ax)
    fig.tight_layout(); sv(fig, 'f2811a.png', 420); NUM['f2811a'] = {k: [round(float(u), 3) for u in v] for k, v in res.items()}
    d = load_diabetes(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=0)
    kw = dict(max_leaf_nodes=4, min_samples_leaf=15, learning_rate=0.05, max_iter=100, random_state=0)
    lo = HGBR(loss='quantile', quantile=0.1, **kw).fit(Xtr, ytr); hi = HGBR(loss='quantile', quantile=0.9, **kw).fit(Xtr, ytr); mid = HGBR(**kw).fit(Xtr, ytr)
    o = np.argsort(mid.predict(Xte)); pl, ph, pm = lo.predict(Xte)[o], hi.predict(Xte)[o], mid.predict(Xte)[o]; yy = yte[o]
    cov = float(((yte >= lo.predict(Xte)) & (yte <= hi.predict(Xte))).mean()); NUM['f2811b'] = {'coverage': round(cov, 3), 'width': round(float((ph - pl).mean()), 1)}
    fig, ax = plt.subplots(figsize=(4.7, 2.7)); i_ = np.arange(len(yy))
    ax.fill_between(i_, pl, ph, color='#BFE3CD', lw=0, label='10% to 90% band', zorder=1); ax.plot(i_, pm, color=GREEN, lw=1.5, label='prediction of the mean', zorder=3)
    ax.scatter(i_, yy, s=9, color=GREY, lw=0, label='measured', zorder=4)
    ax.set_xlabel('test patients, sorted by predicted value'); ax.set_ylabel('progression'); ax.legend(loc='upper left'); grid(ax)
    fig.tight_layout(); sv(fig, 'f2811b.png', 440)
with open('verify/fig_numbers.json', 'w') as fh: json.dump(NUM, fh, indent=1)
with open('verify/fig_textsize.json', 'w') as fh: json.dump(CHK, fh, indent=1)

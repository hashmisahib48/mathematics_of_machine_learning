# Problem figures and answers of Chapter 41.  Run: python3 fig/probs.py
from common import *
from math import comb, log
from sklearn.datasets import load_iris
from sklearn.metrics import (homogeneity_score, completeness_score, fowlkes_mallows_score)
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.labelsize': 9.5, 'legend.fontsize': 9})
A = {}; PCHK = {}


def small(w=2.9, h=1.3): return plt.subplots(figsize=(w, h), dpi=240)


def psv(fig, name, wpx):
    fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig, name)
    with Image.open(name) as im: w, h = im.size
    pw = min(1.55 * wpx, 235 * w / h); PCHK[name] = round(fs * pw / 96 / (w / 240), 1)


def heat(ax, N, rl, cl):
    ax.imshow(N, cmap=matplotlib.colors.LinearSegmentedColormap.from_list('c', ['#FFFFFF', '#9CC0EE']), vmin=0, vmax=N.max() * 1.1)
    for i in range(N.shape[0]):
        for j in range(N.shape[1]):
            ax.text(j, i, str(N[i, j]), ha='center', va='center', fontsize=10, color=NAVY); ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc='none', ec=GRID, lw=0.8))
    ax.set_xticks(range(N.shape[1])); ax.set_xticklabels(cl); ax.set_yticks(range(N.shape[0])); ax.set_yticklabels(rl); ax.tick_params(length=0)
    for s_ in ax.spines.values(): s_.set_visible(False)


# 41A.1 silhouette of one point on a line
pos = {1: [1, 2, 3], 2: [5, 6, 8], 3: [10, 11]}; p = 3
a = np.mean([abs(p - q) for q in (1, 2)]); bs = {k: np.mean([abs(p - q) for q in v]) for k, v in pos.items() if k != 1}; b = min(bs.values())
A['41A.1'] = {'a': float(a), 'b_by_cluster': {k: float(v) for k, v in bs.items()}, 'b': float(b), 's': round(float((b - a) / max(a, b)), 4)}
Xl = np.array([1, 2, 3, 5, 6, 8, 10, 11.]); ll = np.array([0, 0, 0, 1, 1, 1, 2, 2])
assert abs(silhouette_samples(Xl[:, None], ll)[2] - A['41A.1']['s']) < 1e-9
fig, ax = small(2.9, 1.0)
for j in range(3): ax.scatter(Xl[ll == j], np.zeros((ll == j).sum()), s=40, color=COL[j], marker=MARK[j], zorder=3)
ax.scatter([3], [0], s=150, facecolor='none', edgecolor=NAVY, lw=1.2, zorder=4); ax.text(3, 0.5, '$p$', ha='center', fontsize=10, color=NAVY)
ax.set_ylim(-0.4, 0.9); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_position(('data', -0.3)); ax.set_xticks([0, 2, 4, 6, 8, 10, 12]); ax.set_xlim(0, 12.3); ax.set_xlabel('$x$')
psv(fig, 'p_41A1.png', 165)

# 41A.2 Rand and adjusted Rand from a 2 x 2 table
N2 = np.array([[4, 1], [1, 4]]); n = 10
c2 = lambda m: sum(comb(int(v), 2) for v in np.ravel(m))
P11, Pt, Pc = c2(N2), c2(N2.sum(1)), c2(N2.sum(0)); P10, P01 = Pt - P11, Pc - P11; P00 = comb(n, 2) - P11 - P10 - P01
E_ = Pt * Pc / comb(n, 2); ari2 = (P11 - E_) / ((Pt + Pc) / 2 - E_)
tt = np.repeat([0, 1], 5); pp = np.array([0] * 4 + [1] + [0] + [1] * 4)
assert abs(ari(tt, pp) - ari2) < 1e-12
A['41A.2'] = {'P11': P11, 'P10': P10, 'P01': P01, 'P00': P00, 'RI': round((P11 + P00) / comb(n, 2), 4), 'E': round(E_, 4), 'ARI': round(ari2, 4)}
fig, ax = small(1.9, 1.5); heat(ax, N2, ['1', '2'], ['1', '2']); ax.set_xlabel('cluster $j$'); ax.set_ylabel('class $i$'); psv(fig, 'p_41A2.png', 105)

# 41A.3 Calinski-Harabasz and Davies-Bouldin on a line
x6 = np.array([1, 2, 3, 7, 8, 9.]); l6 = np.array([0, 0, 0, 1, 1, 1])
SSB = 3 * (2 - 5) ** 2 + 3 * (8 - 5) ** 2; SSW = 2 + 2; CH = (SSB / 1) / (SSW / 4); S_ = 2 / 3; DB = (S_ + S_) / 6
assert abs(chs(x6[:, None], l6) - CH) < 1e-9 and abs(dbs(x6[:, None], l6) - DB) < 1e-9
A['41A.3'] = {'SSB': SSB, 'SSW': SSW, 'CH': CH, 'S_k': round(S_, 4), 'DB': round(DB, 4)}
fig, ax = small(2.9, 1.0)
for j in range(2): ax.scatter(x6[l6 == j], np.zeros(3), s=40, color=COL[j], marker=MARK[j], zorder=3)
ax.scatter([2, 8], [0, 0], marker='*', s=90, color=GREEN, zorder=4); ax.scatter([5], [0], marker='x', s=40, color=RED, zorder=4)
ax.set_ylim(-0.4, 0.9); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.spines['bottom'].set_position(('data', -0.3)); ax.set_xticks(range(0, 11, 2)); ax.set_xlim(0, 10.3); ax.set_xlabel('$x$')
psv(fig, 'p_41A3.png', 165)

# 41B.1 irises
Xi, yi = load_iris(return_X_y=True); res = {}
lab3 = KMeans(3, n_init=10, random_state=0).fit_predict(Xi)
res['K=3'] = {'ARI': ari(yi, lab3), 'AMI': ami(yi, lab3), 'homogeneity': homogeneity_score(yi, lab3), 'completeness': completeness_score(yi, lab3), 'FM': fowlkes_mallows_score(yi, lab3)}
sil = {}
for K in range(2, 7): sil[K] = silhouette_score(Xi, KMeans(K, n_init=10, random_state=0).fit_predict(Xi))
A['41B.1'] = {'scores_K3': {k: round(float(v), 3) for k, v in res['K=3'].items()}, 'silhouette': {k: round(float(v), 3) for k, v in sil.items()}, 'best_K': max(sil, key=sil.get)}
fig, ax = small(2.9, 1.3); ax.bar(list(sil), list(sil.values()), color=BLUE, width=0.6); ax.set_xlabel('$K$'); ax.set_ylabel('silhouette'); ax.set_ylim(0, 0.8); ax.set_xticks(list(sil)); psv(fig, 'p_41B1.png', 165)

# 41B.2 random labellings: which scores are chance-corrected
from sklearn.metrics import rand_score, normalized_mutual_info_score
yt = np.repeat([0, 1, 2], 50); rg = np.random.default_rng(0); acc = {'RI': [], 'ARI': [], 'NMI': [], 'AMI': []}
for _ in range(20):
    lr = rg.integers(0, 30, 150)
    acc['RI'].append(rand_score(yt, lr)); acc['ARI'].append(ari(yt, lr)); acc['NMI'].append(normalized_mutual_info_score(yt, lr)); acc['AMI'].append(ami(yt, lr))
A['41B.2'] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
fig, ax = small(2.9, 1.3); ax.bar(list(acc), [np.mean(v) for v in acc.values()], color=[GREY, BLUE, GREY, BLUE], width=0.6); ax.set_ylim(0, 0.8); ax.set_ylabel('mean score'); psv(fig, 'p_41B2.png', 165)

# 41C.1 raw wine
Xw, yw = load_wine(return_X_y=True); Xs = StandardScaler().fit_transform(Xw); C1 = {}
for nm_, Z in (('raw', Xw), ('standardised', Xs)):
    lab = KMeans(3, n_init=10, random_state=0).fit_predict(Z); C1[nm_] = {'silhouette': round(float(silhouette_score(Z, lab)), 3), 'ARI': round(float(ari(yw, lab)), 3)}
A['41C.1'] = C1
fig, ax = small(2.9, 1.4); xs_ = np.arange(2)
ax.bar(xs_ - 0.19, [C1[k]['silhouette'] for k in C1], 0.36, color=BLUE, label='silhouette'); ax.bar(xs_ + 0.19, [C1[k]['ARI'] for k in C1], 0.36, color=GREY, label='ARI')
ax.set_xticks(xs_); ax.set_xticklabels(list(C1)); ax.set_ylim(0, 1.12); ax.legend(loc='upper left', ncol=2, fontsize=9, handlelength=1, columnspacing=0.8, borderaxespad=0.1); psv(fig, 'p_41C1.png', 165)

# 41C.2 a stable but wrong K
lab2 = KMeans(2, n_init=10, random_state=0).fit_predict(XB)
A['41C.2'] = {'ARI_K2': round(float(ari(YB, lab2)), 3), 'silhouette_K2': round(float(silhouette_score(XB, lab2)), 3)}
fig, ax = small(2.4, 1.5); pts(ax, XB, lab2, s=4); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('four blobs, $K$ = 2'); psv(fig, 'p_41C2.png', 135)

dump('problem_answers.json', A); dump('probfig_textsize.json', PCHK)
print(json.dumps(A, indent=0)[:2500]); print(PCHK)

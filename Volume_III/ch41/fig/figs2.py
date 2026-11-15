# Figures 41.3-1, 41.4-1, 41.9-1.  Run: python3 fig/figs2.py
from common import *
from math import comb

# ---- 41.3-1: silhouette plots for K = 3, 4, 5 on the four blobs
fig, axs = plt.subplots(1, 3, figsize=(5.4, 2.3), dpi=240, sharex=True)
NUM = {'sil_plot': {}}
for ax, K in zip(axs, (3, 4, 5)):
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(XB)
    s = silhouette_samples(XB, lab); mean = s.mean(); y0 = 0; ticks = []
    order = np.argsort([-s[lab == j].mean() for j in range(K)])
    per = {}
    for r, j in enumerate(order):
        v = np.sort(s[lab == j])[::-1]
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=COL[j % 6], lw=0, alpha=0.9)
        ticks.append(y0 + len(v) / 2); per[int(r + 1)] = {'n': int(len(v)), 'mean': round(float(v.mean()), 3), 'min': round(float(v.min()), 3)}
        y0 += len(v) + 8
    ax.axvline(mean, color=RED, lw=1.2, ls='--'); ax.set_yticks(ticks); ax.set_yticklabels([str(i + 1) for i in range(K)])
    ax.set_xlim(-0.25, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.set_title(f'$K$ = {K}:  mean {mean:.2f}', fontsize=9, color=NAVY)
    ax.set_xlabel('silhouette $s(i)$'); NUM['sil_plot'][K] = {'mean': round(float(mean), 3), 'negative': int((s < 0).sum()), 'clusters': per}
axs[0].set_ylabel('cluster')
fig.tight_layout(w_pad=0.5); sv(fig, 'f431.png', 470)

# ---- 41.4-1: the 28 pairs of the eight customers
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.45), dpi=240, gridspec_kw={'width_ratios': [1.25, 1]})
ax = axs[0]
colours = {(1, 1): GREEN, (1, 0): OR, (0, 1): RED, (0, 0): LGREY}
cnt = {k: 0 for k in colours}
for p in range(8):
    for q in range(p + 1, 8):
        k = (int(SEG[p] == SEG[q]), int(CLU[p] == CLU[q])); cnt[k] += 1
        ax.add_patch(plt.Rectangle((q - .5, 7 - p - .5), 1, 1, fc=colours[k], ec='white', lw=1.2))
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 7.5); ax.set_aspect('equal')
ax.set_xticks(range(8)); ax.set_xticklabels(NAMES); ax.set_yticks(range(8)); ax.set_yticklabels(NAMES[::-1])
ax.xaxis.tick_top(); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
NUM['pairs'] = {'P11': cnt[(1, 1)], 'P10': cnt[(1, 0)], 'P01': cnt[(0, 1)], 'P00': cnt[(0, 0)]}
ax = axs[1]; ax.axis('off'); ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.2)
ax.text(2.3, 4.0, 'together in the clusters?', ha='center', fontsize=8.5, color=NAVY)
ax.text(1.55, 3.6, 'yes', ha='center', fontsize=8.5, color=NAVY); ax.text(3.05, 3.6, 'no', ha='center', fontsize=8.5, color=NAVY)
ax.text(0.9, 2.55, 'yes', ha='right', va='center', fontsize=8.5, color=NAVY); ax.text(0.9, 1.05, 'no', ha='right', va='center', fontsize=8.5, color=NAVY)
ax.text(0.15, 1.8, 'together in\nthe segments?', ha='center', va='center', fontsize=8.5, color=NAVY, rotation=90)
for (r_, c_), key, nm in (((2.0, 0.95), (1, 1), 'P11'), ((2.0, 2.45), (1, 0), 'P10'), ((0.5, 0.95), (0, 1), 'P01'), ((0.5, 2.45), (0, 0), 'P00')):
    pass
cells = {(1, 1): (1.0, 2.0), (1, 0): (2.5, 2.0), (0, 1): (1.0, 0.5), (0, 0): (2.5, 0.5)}
nm = {(1, 1): 'P11', (1, 0): 'P10', (0, 1): 'P01', (0, 0): 'P00'}
# rows: truth together (1) top; columns: clustering together (1) left.  key = (segments, clusters)
for key, (x0, y0) in cells.items():
    col = colours[key]
    ax.add_patch(plt.Rectangle((x0 + 0.05, y0 + 0.05), 1.4, 1.4, fc=col, ec='white', lw=1.0, alpha=1.0))
    ax.text(x0 + 0.75, y0 + 0.75, f'{nm[key][0]}$_{{{nm[key][1:]}}}$ = {cnt[key]}', ha='center', va='center', fontsize=9.5, color='white' if key != (0, 0) else NAVY, weight='bold')
fig.tight_layout(w_pad=0.5); sv(fig, 'f441.png', 450)

# ---- 41.9-1: chance behaviour of RI, NMI, ARI, AMI
rng = np.random.default_rng(0); n = 150; truth = np.repeat([0, 1, 2], 50); reps = 20
Ks = [2, 3, 4, 6, 8, 12, 16, 24, 32]; names = ['ARI', 'AMI', 'RI', 'NMI']
fns = {'ARI': ari, 'AMI': ami, 'RI': rand_score, 'NMI': nmi}
R1 = {nm_: [] for nm_ in names}
for K in Ks:
    acc = {nm_: [] for nm_ in names}
    for _ in range(reps):
        lab = rng.integers(0, K, n)
        for nm_ in names: acc[nm_].append(fns[nm_](truth, lab))
    for nm_ in names: R1[nm_].append(float(np.mean(acc[nm_])))
fr = np.linspace(0, 1, 11); R2 = {nm_: [] for nm_ in names}
for f in fr:
    acc = {nm_: [] for nm_ in names}
    for _ in range(reps):
        lab = truth.copy(); m = rng.random(n) < f; lab[m] = rng.integers(0, 3, m.sum())
        for nm_ in names: acc[nm_].append(fns[nm_](truth, lab))
    for nm_ in names: R2[nm_].append(float(np.mean(acc[nm_])))
sty = {'ARI': (BLUE, '-', 'o'), 'AMI': (TEAL, '-', '^'), 'RI': (RED, '--', 's'), 'NMI': (OR, '--', 'D')}
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.3), dpi=240)
for nm_ in names:
    c_, ls_, mk_ = sty[nm_]
    axs[0].plot(Ks, R1[nm_], color=c_, ls=ls_, marker=mk_, ms=3.5, lw=1.3, label=nm_)
    axs[1].plot(fr, R2[nm_], color=c_, ls=ls_, marker=mk_, ms=3.5, lw=1.3, label=nm_)
axs[0].set_xlabel('number of random clusters $K$'); axs[0].set_ylabel('mean score'); axs[0].set_ylim(-0.05, 1.02)
axs[1].set_xlabel('share of labels redrawn at random'); axs[1].set_ylim(-0.05, 1.02); axs[1].set_yticklabels([])
axs[0].legend(loc='center right', bbox_to_anchor=(1.0, 0.42), ncol=1, fontsize=8.5, handlelength=2.0, borderaxespad=0.2, labelspacing=0.2)
for ax in axs: grid(ax)
fig.tight_layout(w_pad=0.7); sv(fig, 'f491.png', 470)
NUM['chance'] = {'Ks': Ks, **{k: [round(v, 3) for v in R1[k]] for k in names}}
NUM['damage'] = {'share': [round(float(x), 1) for x in fr], **{k: [round(v, 3) for v in R2[k]] for k in names}}
dump('fig_numbers2.json', NUM); dump('fig_textsize2.json', CHK)

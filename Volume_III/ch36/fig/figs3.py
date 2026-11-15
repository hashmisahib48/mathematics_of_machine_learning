# Hyperparameter pictures, failures, quantisation and case studies. Run: python3 fig/figs3.py
from common import *
import runpy, io, contextlib
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, load_wine, load_digits
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---- f3611a: the knob K
cent = [(-5, -3), (-3, 4), (4, -4), (5, 3)]
Xg, yg = make_blobs(400, centers=cent, cluster_std=1.5, random_state=3)
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.75), dpi=240, sharex=True, sharey=True)
Jg = {}
for ax, K in zip(axs, (2, 3, 4, 6)):
    km = KMeans(K, n_init=10, random_state=0).fit(Xg); Jg[K] = round(float(km.inertia_), 1)
    voronoi_shade(ax, km.cluster_centers_, (-10, 10), (-9, 9), cols=SHADE, n=200)
    pts(ax, Xg, km.labels_, s=5); centres(ax, km.cluster_centers_, 42)
    ax.set_xlim(-10, 10); ax.set_ylim(-9, 9); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'$K$ = {K}', fontsize=9, color=NAVY)
    ax.set_xlabel(f'$J$ = {km.inertia_:.0f}', fontsize=9)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
NUM['gallery_J'] = Jg
fig.tight_layout(w_pad=0.3); sv(fig, 'f3611a.png', 470)

# ---- f3611b: elbow, silhouette, ARI
Ks = list(range(1, 10)); J = []; S = {}; A = {}
for K in Ks:
    km = KMeans(K, n_init=10, random_state=0).fit(Xg); J.append(float(km.inertia_))
    if K > 1:
        S[K] = float(silhouette_score(Xg, km.labels_)); A[K] = float(adjusted_rand_score(yg, km.labels_))
NUM['elbow'] = {'J': [round(v, 1) for v in J], 'sil': {k: round(v, 3) for k, v in S.items()}, 'ari': {k: round(v, 3) for k, v in A.items()}}
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.1), dpi=240)
ax = axs[0]
ax.plot(Ks, J, '-o', color=RED, ms=4, lw=1.3); ax.axvline(4, color=PUR, lw=1.1, ls='--')
ax.set_xlabel('$K$'); ax.set_ylabel('inertia $J$'); ax.set_ylim(0, None); grid(ax); ax.set_xticks(Ks)
ax = axs[1]
ax.plot(list(S), list(S.values()), '-o', color=BLUE, ms=4, lw=1.3, label='silhouette')
ax.plot(list(A), list(A.values()), '--s', color=GREY, ms=4, lw=1.3, label='ARI')
ax.axvline(4, color=PUR, lw=1.1, ls='--'); ax.set_xlabel('$K$'); ax.set_ylim(0, 1.05); ax.set_xticks(Ks); grid(ax)
ax.legend(loc='upper right', fontsize=8.5)
fig.tight_layout(w_pad=1.0); sv(fig, 'f3611b.png', 450)

# ---- f3612a: where k-means fails
rng = np.random.RandomState(170)
Xa, ya = make_blobs(450, centers=3, random_state=170)
Xa = Xa @ np.array([[0.6, -0.64], [-0.4, 0.85]])
Xv, yv = make_blobs(450, centers=3, cluster_std=[1.0, 2.7, 0.5], random_state=170)
Xs, ys = make_blobs([450, 60, 12], centers=[(-4, 0), (3, 0), (0, 4)], cluster_std=[1.2, 1.2, 0.6], random_state=5)
Xm, ym = make_moons(400, noise=0.07, random_state=0)
cases = [('stretched', Xa, ya, 3), ('unequal spread', Xv, yv, 3), ('unequal size', Xs, ys, 3), ('two moons', Xm, ym, 2)]
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.85), dpi=240)
NUM['fail'] = {}
for ax, (nm, Xf, yf, K) in zip(axs, cases):
    km = KMeans(K, n_init=10, random_state=0).fit(Xf); ari = adjusted_rand_score(yf, km.labels_)
    NUM['fail'][nm] = {'ARI': round(float(ari), 3), 'sizes': np.bincount(km.labels_).tolist(), 'true_sizes': np.bincount(yf).tolist()}
    pts(ax, Xf, km.labels_, s=7, truth=yf); centres(ax, km.cluster_centers_, 50)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(nm, fontsize=9, color=NAVY)
    ax.set_xlabel(f'ARI = {ari:.2f}', fontsize=9)
    ax.spines['top'].set_visible(True); ax.spines['right'].set_visible(True)
fig.tight_layout(w_pad=0.4); sv(fig, 'f3612a.png', 470)

# ---- f3612b: quantisation
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path('code/quantise.py')
img, quant = ns['img'], ns['quant']
fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.55), dpi=240)
for ax, (t, im) in zip(axs, [('original', img), ('$K$ = 4', quant[4]), ('$K$ = 8', quant[8]), ('$K$ = 16', quant[16])]):
    ax.imshow(np.clip(im, 0, 255).astype(np.uint8)); ax.axis('off'); ax.set_title(t, fontsize=9, color=NAVY)
fig.tight_layout(w_pad=0.3); sv(fig, 'f3612b.png', 470)

# ---- f3613a: wine, raw and standardised
wine = load_wine(); Xr = wine.data; cult = wine.target; Xz = StandardScaler().fit_transform(Xr)
names = list(wine.feature_names); ix, iy = names.index('flavanoids'), names.index('proline')
fig, axs = plt.subplots(1, 2, figsize=(4.9, 2.2), dpi=240, sharey=True)
for ax, (t, Xk) in zip(axs, (('raw features', Xr), ('standardised features', Xz))):
    km = KMeans(3, n_init=10, random_state=0).fit(Xk)
    pts(ax, np.c_[Xr[:, ix], Xr[:, iy]], km.labels_, s=14, truth=cult)
    ax.set_title(f'{t}: ARI = {adjusted_rand_score(cult, km.labels_):.2f}', fontsize=8.8, color=NAVY)
    ax.set_xlabel('flavanoids'); grid(ax)
axs[0].set_ylabel('proline')
fig.tight_layout(w_pad=0.6); sv(fig, 'f3613a.png', 450)

# ---- f3613b: digits, the ten centres
dg = load_digits(); km = KMeans(10, n_init=10, random_state=0).fit(dg.data)
fig, axs = plt.subplots(2, 5, figsize=(4.9, 2.6), dpi=240)
for k, ax in enumerate(axs.ravel()):
    ax.imshow(km.cluster_centers_[k].reshape(8, 8), cmap='gray_r', vmin=0, vmax=16); ax.set_xticks([]); ax.set_yticks([])
    m = km.labels_ == k; mj = np.bincount(dg.target[m], minlength=10)
    ax.set_title(f'{m.sum()} · digit {mj.argmax()}', fontsize=8.8, color=NAVY)
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_color(GREEN); sp_.set_linewidth(1.0)
fig.tight_layout(h_pad=1.2, w_pad=0.3); sv(fig, 'f3613b.png', 450)

json.dump(NUM, open('verify/fig_numbers_cases.json', 'w'), indent=1)
json.dump(CHK, open('verify/fig_textsize_cases.json', 'w'), indent=1)
print(json.dumps(NUM)); print({k: v['min_font_printed'] for k, v in CHK.items()})

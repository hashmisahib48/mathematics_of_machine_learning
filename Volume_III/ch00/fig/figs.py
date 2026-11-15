# Figures of the refresher. Run: python3 fig/figs.py (PNGs into ch00/, numbers into verify/fig_numbers.json)
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from matplotlib.patches import Ellipse
from sklearn.datasets import load_digits
NUM = {}

# ------------------------------------------------ R.0-1  map of the refresher
fig, ax = plt.subplots(figsize=(5.0, 2.6), dpi=240)
ax.set_xlim(0, 10); ax.set_ylim(0, 5.2); ax.axis('off')
items = [('R.1', 'Notation and the API', 'every chapter', LB, BLUE),
         ('R.2', 'Linear algebra, SVD', 'Ch. 42 to 45', '#E1F2E8', GREEN),
         ('R.3', 'Calculus and optimisation', 'Ch. 40, 43, 44', '#FDEBD0', OR),
         ('R.4', 'Probability and KL', 'Ch. 40, 41, 44, 53', PINK, RED),
         ('R.5', 'Distances and kernels', 'Ch. 36 to 38, 44, 45', LB, BLUE),
         ('R.6', 'The supervised models', 'Ch. 49 to 59', '#E1F2E8', GREEN),
         ('R.7', 'Generalisation, evaluation', 'Ch. 54 to 58', '#FDEBD0', OR),
         ('R.8', 'Graphs and the Laplacian', 'Ch. 39, 44', PINK, RED)]
for k, (tag, t1, t2, fc, ec) in enumerate(items):
    x0 = 0.05 + (k // 4) * 5.0; y0 = 4.1 - (k % 4) * 1.3
    ax.add_patch(FancyBboxPatch((x0, y0), 4.8, 1.05, boxstyle='round,pad=0.02,rounding_size=0.1', fc=fc, ec=ec, lw=1.0))
    ax.text(x0 + 0.15, y0 + 0.74, '§' + tag + '  ' + t1, ha='left', va='center', fontsize=9.4, weight='bold', color=NAVY)
    ax.text(x0 + 0.15, y0 + 0.28, t2, ha='left', va='center', fontsize=9.4, color=GREY)
save(fig, 'f000.png')

# ------------------------------------------------ R.2-1  SVD: axes of a cloud, and the Eckart-Young bars
rng = np.random.default_rng(3)
Z = rng.normal(size=(40, 2)) * [2.2, 0.7]
th = np.deg2rad(30); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Xc = Z @ R.T; Xc -= Xc.mean(0)
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.2), dpi=240, gridspec_kw={'width_ratios': [1.05, 1]})
ax = axs[0]
ax.scatter(Xc[:, 0], Xc[:, 1], s=9, color=BLUE, zorder=3)
v1, v2 = Vt[0], Vt[1]
proj = np.outer(Xc @ v1, v1)
for k in range(0, 40, 3): ax.plot([Xc[k, 0], proj[k, 0]], [Xc[k, 1], proj[k, 1]], color=RED, lw=0.8, zorder=2)
L1 = s[0] / np.sqrt(39) * 2; L2 = s[1] / np.sqrt(39) * 2
arr(ax, (0, 0), tuple(v1 * L1), c=GREEN, lw=1.6, ms=9); arr(ax, (0, 0), tuple(v2 * L2), c=GREEN, lw=1.6, ms=9)
ax.text(*(v1 * (L1 + 0.9) + np.array([-0.2, -0.2])), '$v_1$', color=GREEN, fontsize=10)
ax.text(*(v2 * (L2 + 0.5) + np.array([-0.2, 0.0])), '$v_2$', color=GREEN, fontsize=10)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('centred cloud and its axes', fontsize=9.8, color=NAVY)
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax = axs[1]
dg = load_digits().images[3].astype(float); sv = np.linalg.svd(dg, compute_uv=False)
k = 2; e = sv ** 2
ax.bar(np.arange(1, 9)[:k], e[:k], color=GREEN, label='kept, $k=2$'); ax.bar(np.arange(1, 9)[k:], e[k:], color=RED, label='discarded')
ax.set_xlabel('index $j$', fontsize=9); ax.set_ylabel('$\\sigma_j^{\\,2}$', fontsize=9.5); ax.tick_params(labelsize=9.5)
ax.set_xticks(range(1, 9)); ax.legend(fontsize=9.5, loc='upper right'); ax.set_title('an 8 × 8 digit image', fontsize=9.8, color=NAVY)
NUM['digit_total'] = float(e.sum()); NUM['digit_disc'] = float(e[k:].sum()); NUM['digit_rel'] = float(e[k:].sum() / e.sum())
NUM['cloud_sigma'] = s.tolist()
fig.tight_layout(); save(fig, 'f021.png')

# ------------------------------------------------ R.4-1  KL integrand, and a bivariate normal
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.2), dpi=240, gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
xs = np.linspace(-4, 6, 600)
nq = lambda x, m, v: np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
q = nq(xs, 0, 1); g = nq(xs, 1, 4); integ = q * np.log(q / g)
ax.plot(xs, q, color=BLUE, lw=1.4, label='$q=N(0,1)$'); ax.plot(xs, g, color=OR, lw=1.4, ls='--', label='$\\gamma=N(1,4)$')
ax.fill_between(xs, 0, integ, color=RED, alpha=0.35, label='$q\\,\\ln(q/\\gamma)$')
ax.axhline(0, color=GREY, lw=0.5); ax.set_xlim(-4, 6); ax.set_ylim(-0.12, 0.5)
ax.legend(fontsize=9.5, loc='upper right'); ax.tick_params(labelsize=9.5); ax.set_xlabel('$x$', fontsize=9.5)
NUM['kl_numeric'] = float(np.trapezoid(integ, xs))
ax = axs[1]
mu = np.array([1., 2]); Sg = np.array([[4., 2], [2, 3]])
lam, Q = np.linalg.eigh(Sg)
for c_ in (1, 2):
    ax.add_patch(Ellipse(mu, 2 * c_ * np.sqrt(lam[1]), 2 * c_ * np.sqrt(lam[0]), angle=np.degrees(np.arctan2(Q[1, 1], Q[0, 1])), fc='none', ec=GREEN, lw=1.2))
for j in (0, 1): arr(ax, tuple(mu), tuple(mu + Q[:, j] * np.sqrt(lam[j])), c=NAVY, lw=1.2, ms=8)
ax.scatter([3], [3], color=RED, s=22, zorder=4); ax.text(3.15, 3.15, '$\\mathbf{x}$', color=RED, fontsize=10)
ax.scatter(*mu, color=NAVY, s=10, zorder=4)
ax.set_xlim(-3.5, 5.5); ax.set_ylim(-2, 6.5); ax.set_aspect('equal'); ax.tick_params(labelsize=9.5)
ax.set_xlabel('$x_1$', fontsize=9.5); ax.set_ylabel('$x_2$', fontsize=9.5)
fig.tight_layout(); save(fig, 'f041.png')

# ------------------------------------------------ R.7-1  validation curve and k-fold diagram
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(1)
xd = np.sort(rng.uniform(0, 1, 30)); yd = np.sin(2 * np.pi * xd) + rng.normal(0, 0.3, 30)
degs = range(1, 10); tr_, cv_ = [], []
for d in degs:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()); m.fit(xd[:, None], yd)
    tr_.append(float(np.mean((yd - m.predict(xd[:, None])) ** 2)))
    cv_.append(float(-cross_val_score(m, xd[:, None], yd, cv=5, scoring='neg_mean_squared_error').mean()))
NUM['val_deg'] = list(degs); NUM['val_train'] = tr_; NUM['val_cv'] = cv_; NUM['val_best'] = int(list(degs)[int(np.argmin(cv_))])
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.1), dpi=240, gridspec_kw={'width_ratios': [1.1, 1]})
ax = axs[0]
ax.plot(list(degs), tr_, color=BLUE, marker='o', ms=3.5, lw=1.3, label='training'); ax.plot(list(degs), cv_, color=OR, marker='s', ms=3.5, ls='--', lw=1.3, label='5-fold CV')
ax.axvline(NUM['val_best'], color=PUR, lw=0.9, ls=':'); ax.set_yscale('log')
ax.set_xlabel('polynomial degree', fontsize=9.5); ax.set_ylabel('mean squared error', fontsize=9.5); ax.tick_params(labelsize=9.5); ax.legend(fontsize=9.5, loc='upper left')
ax = axs[1]; ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-0.4, 6)
ax.text(5, 5.6, '4-fold cross-validation', ha='center', fontsize=9.8, color=NAVY)
for r_ in range(4):
    for c_ in range(4):
        ax.add_patch(Rectangle((0.6 + c_ * 2, 4.1 - r_ * 1.05), 1.9, 0.85, fc=OR if r_ == c_ else LB, ec=BLUE, lw=0.7))
    ax.text(0.45, 4.52 - r_ * 1.05, f'{r_ + 1}', ha='right', va='center', fontsize=9.5, color=GREY)
ax.add_patch(Rectangle((0.6, -0.05), 1.9, 0.5, fc=OR, ec=BLUE, lw=0.7)); ax.text(2.7, 0.2, 'scored', fontsize=9.5, va='center', color=NAVY)
ax.add_patch(Rectangle((5.0, -0.05), 1.9, 0.5, fc=LB, ec=BLUE, lw=0.7)); ax.text(7.1, 0.2, 'fitted', fontsize=9.5, va='center', color=NAVY)
fig.tight_layout(); save(fig, 'f071.png')

# ------------------------------------------------ R.8-1  graphs and the second eigenvector
def lap(E, n=5):
    W = np.zeros((n, n))
    for i, j in E: W[i, j] = W[j, i] = 1
    return np.diag(W.sum(1)) - W
two = [(0, 1), (1, 2), (0, 2), (3, 4)]; br = two + [(2, 3)]
pos = np.array([[0, 0.9], [0.9, 1.5], [1.3, 0.2], [2.9, 0.2], [3.7, 1.0]])
ev, V = np.linalg.eigh(lap(br)); f = V[:, 1]; f = f * (1 if f[0] < 0 else -1)
NUM['fiedler'] = f.tolist(); NUM['lap_bridge_eig'] = ev.tolist()
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.75), dpi=240)
for ax, E, ttl, col in ((axs[0], two, 'two components: $\\lambda=0,0,2,3,3$', None), (axs[1], br, 'one bridge: $\\lambda_2=%.2f$' % ev[1], f)):
    for i, j in E: ax.plot(*pos[[i, j]].T, color=GREY, lw=1.6 if (i, j) != (2, 3) else 1.6, zorder=1, ls='-' if (i, j) != (2, 3) else '--')
    cs = [BLUE] * 5 if col is None else [OR if v > 0 else BLUE for v in col]
    ax.scatter(pos[:, 0], pos[:, 1], s=95, c=cs, edgecolor=NAVY, zorder=2)
    for k_, (x_, y_) in enumerate(pos): ax.text(x_, y_, str(k_ + 1), ha='center', va='center', fontsize=9.5, color='white' if col is None or col[k_] <= 0 else NAVY, zorder=3)
    if col is not None:
        for k_, (x_, y_) in enumerate(pos): ax.text(x_, y_ + (0.42 if k_ == 1 else -0.42), '%+.2f' % col[k_], ha='center', va='center', fontsize=9.8, color=NAVY)
    ax.set_xlim(-0.4, 4.2); ax.set_ylim(-0.6, 1.9); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(ttl, fontsize=9.8, color=NAVY)
fig.tight_layout(); save(fig, 'f081.png')
json.dump(NUM, open('verify/fig_numbers.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in NUM.items() if k not in ('val_train', 'val_cv', 'val_deg')}))
print('val', [round(a, 3) for a in NUM['val_cv']], [round(a, 3) for a in NUM['val_train']])

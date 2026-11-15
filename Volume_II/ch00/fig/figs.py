# All figures of the refresher. Run: python3 fig/figs.py  (writes PNGs into ch00/ and numbers into verify/fig_numbers.json)
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from matplotlib.patches import Ellipse
NUM = {}

# ---------------------------------------------------------------- R.0-1  the loop of supervised learning
fig, ax = plt.subplots(figsize=(5.4, 2.1), dpi=240)
ax.set_xlim(0, 10.8); ax.set_ylim(0, 5.2); ax.axis('off')
def bx(x, y, w, h, t1, t2, fc, ec, fs=8.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h * 0.66, t1, ha='center', va='center', fontsize=fs, weight='bold', color=NAVY)
    ax.text(x + w / 2, y + h * 0.28, t2, ha='center', va='center', fontsize=fs, color=NAVY)
xs = [0.1, 2.3, 4.5, 6.7, 8.9]; W = 1.8; Y0 = 3.15; Hh = 1.2
spec = [('Data', 'X, y', LB, BLUE, '§R.1'), ('Model', 'ŷ = f(x)', '#E1F2E8', GREEN, '§R.7'),
        ('Loss', 'J(w, b)', PINK, RED, '§R.7, R.8'), ('Optimiser', 'min J', '#FDEBD0', OR, '§R.3, R.5'),
        ('Fitted model', 'w, b', '#E1F2E8', GREEN, '§R.1')]
for x0, (a, b_, fc, ec, tag) in zip(xs, spec):
    bx(x0, Y0, W, Hh, a, b_, fc, ec)
    ax.text(x0 + W / 2, Y0 + Hh + 0.2, tag, ha='center', va='center', fontsize=9, color=GREY)
for x0 in xs[:-1]: arr(ax, (x0 + W + 0.04, Y0 + Hh / 2), (x0 + 2.2 - 0.02, Y0 + Hh / 2), c=NAVY, lw=1.1, ms=8)
# evaluation on new data, with feedback to the settings
bx(6.7, 1.2, 4.0, 1.15, 'Judge on new data', 'split, cross-validation, scores', LGREY, GREY, fs=8.9)
ax.text(8.7, 0.88, '§R.6, R.8', ha='center', va='center', fontsize=9, color=GREY)
arr(ax, (9.8, Y0 - 0.03), (9.8, 2.38), c=NAVY, lw=1.1, ms=8)
ax.annotate('', xy=(3.2, Y0 - 0.03), xytext=(6.65, 1.78), arrowprops=dict(arrowstyle='-|>', color=PUR, lw=1.1, connectionstyle='angle,angleA=0,angleB=90,rad=0', mutation_scale=8))
ax.text(4.95, 2.02, 'choose the settings α, C', ha='center', va='center', fontsize=9.2, color=PUR, style='italic')
# the language underneath
for k, (t, tag) in enumerate([('Linear algebra', '§R.2'), ('Calculus', '§R.3'), ('Probability', '§R.4')]):
    x0 = 0.1 + k * 2.4
    ax.add_patch(FancyBboxPatch((x0, 0.15), 2.25, 0.75, boxstyle='round,pad=0.02,rounding_size=0.1', fc='white', ec=GREY, lw=0.8, ls='--'))
    ax.text(x0 + 1.125, 0.60, t, ha='center', va='center', fontsize=9, color=NAVY)
    ax.text(x0 + 1.125, 0.26, tag, ha='center', va='center', fontsize=9, color=GREY)
ax.text(0.1, 1.2, 'the language of all of the above', ha='left', va='center', fontsize=9.2, color=GREY, style='italic')
save(fig, 'f000.png')

# ---------------------------------------------------------------- R.2-1  unit balls and a projection
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.1), dpi=240, gridspec_kw={'width_ratios': [1, 1.1]})
ax = axs[0]; t = np.linspace(0, 2 * np.pi, 400)
ax.fill([-1, 0, 1, 0], [0, 1, 0, -1], color=PINK, alpha=0.0)
ax.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color=PUR, lw=1.7, ls='--', label='q = 1')
ax.plot(np.cos(t), np.sin(t), color=BLUE, lw=1.7, label='q = 2')
ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=GREY, lw=1.4, ls=':', label='q = ∞')
ax.axhline(0, color=GRID, lw=0.6); ax.axvline(0, color=GRID, lw=0.6)
ax.set_aspect('equal'); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.tick_params(labelsize=8.5)
ax.set_xlabel('$x_1$', fontsize=9.5); ax.set_ylabel('$x_2$', fontsize=9.5)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), fontsize=9, ncol=3, handlelength=1.5, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
ax.set_title('‖x‖$_q$ = 1', fontsize=9.5, color=NAVY, pad=4)
ax.set_ylim(-1.45, 1.75)
ax = axs[1]
a = np.array([2., 1.]); b = np.array([3., 4.]); p = 2 * a; r = b - p
ax.plot([0, 5.0], [0, 2.5], color=GRID, lw=0.9, zorder=1)
arr(ax, (0, 0), tuple(a), c=BLUE, lw=1.6, ms=8); arr(ax, (0, 0), tuple(b), c=BLUE, lw=1.6, ms=8)
arr(ax, (0, 0), tuple(p), c=GREEN, lw=2.0, ms=8); arr(ax, tuple(p), tuple(b), c=RED, lw=1.6, ms=8)
s = 0.28; u = a / np.linalg.norm(a); v = r / np.linalg.norm(r)
sq = np.array([p - s * u, p - s * u + s * v, p + s * v])
ax.plot(sq[:, 0], sq[:, 1], color=NAVY, lw=0.8)
ax.text(1.0, 0.12, 'a', color=BLUE, fontsize=10, style='italic'); ax.text(3.2, 4.0, 'b', color=BLUE, fontsize=10, style='italic')
ax.text(4.15, 1.75, 'p = 2a', color=GREEN, fontsize=9.5, weight='bold'); ax.text(2.9, 2.7, 'r = b − p', color=RED, fontsize=9.5, ha='right')
ax.text(4.2, 3.2, 'a·r = 0', color=NAVY, fontsize=9)
ax.set_aspect('equal'); ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.4, 4.6); ax.tick_params(labelsize=8.5)
ax.set_xlabel('$x_1$', fontsize=9.5); ax.set_ylabel('$x_2$', fontsize=9.5)
ax.set_title('projection of b on a', fontsize=9.5, color=NAVY, pad=4)
fig.tight_layout(w_pad=1.0); save(fig, 'f021.png')

# ---------------------------------------------------------------- R.3-1  gradient arrows on contours (delivery loss)
X = np.c_[np.ones(5), np.arange(1., 6)]; y = np.array([3., 4, 6, 7, 10])
J = lambda w: np.sum((y - X @ w) ** 2)
grad = lambda w: -2 * X.T @ (y - X @ w)
bb, ww = np.meshgrid(np.linspace(-1.2, 3.0, 200), np.linspace(-0.2, 2.9, 200))
Jg = ((y[None, None, :] - bb[..., None] - ww[..., None] * np.arange(1., 6)[None, None, :]) ** 2).sum(-1)
fig, ax = plt.subplots(figsize=(4.7, 2.95), dpi=240)
lv = [1.5, 4, 10, 25, 60, 130, 250]
ax.contourf(bb, ww, Jg, levels=[1.1] + lv, cmap='Blues', alpha=0.8, extend='max')
ax.contour(bb, ww, Jg, levels=lv, colors='#1F5FBF', linewidths=0.6)
gb, gw = np.meshgrid(np.linspace(-0.8, 2.6, 5), np.linspace(0.5, 2.6, 4))
for b0, w0 in zip(gb.ravel(), gw.ravel()):
    g = -grad(np.array([b0, w0])); g = g / np.linalg.norm(g) * 0.32
    arr(ax, (b0, w0), (b0 + g[0], w0 + g[1]), c=RED, lw=1.1, ms=6)
path = [np.zeros(2)]
for _ in range(30): path.append(path[-1] - 0.01 * grad(path[-1]))
path = np.array(path); NUM['gd_J'] = [round(float(J(w)), 3) for w in path[:6]]; NUM['gd_end'] = path[-1].round(4).tolist()
ax.plot(path[:, 0], path[:, 1], '-o', color=NAVY, lw=0.9, ms=2.4, zorder=5)
ax.annotate('', xy=(0.9, 1.7), xytext=(0, 0), arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=1.5, ls='--', mutation_scale=9, shrinkA=0, shrinkB=2), zorder=6)
ax.scatter([0.9], [1.7], marker='*', s=90, color=GREEN, edgecolor='white', lw=0.6, zorder=7)
ax.text(1.0, 1.95, 'minimum (0.9, 1.7)', color=GREEN, fontsize=9, weight='bold', va='bottom')
ax.text(0.12, -0.05, 'start (0, 0)', color=NAVY, fontsize=9, va='center')
ax.text(2.95, 0.45, 'red: −∇J (direction)\nnavy: 30 gradient steps, η = 0.01\ngreen: one Newton step', fontsize=8.3, ha='right', va='center', color=NAVY, bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))
ax.set_xlabel('intercept  $b$', fontsize=9.5); ax.set_ylabel('slope  $w$', fontsize=9.5); ax.tick_params(labelsize=8.5)
ax.set_aspect('equal'); ax.set_xlim(-1.2, 3.0); ax.set_ylim(-0.2, 2.9); save(fig, 'f031.png')

# ---------------------------------------------------------------- R.4-1  conditioning a bivariate normal
mu = np.array([1., 2.]); S = np.array([[4., 2.], [2., 3.]]); xa = 3.0
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.1), dpi=240, gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]; lam, Q = np.linalg.eigh(S)
for k, ls in ((1, '-'), (2, '--')):
    ang = np.degrees(np.arctan2(Q[1, 1], Q[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * k * np.sqrt(lam[1]), 2 * k * np.sqrt(lam[0]), angle=ang, fill=False, ec=BLUE, lw=1.3, ls=ls))
xx = np.array([-3.5, 6.0]); ax.plot(xx, mu[1] + S[0, 1] / S[0, 0] * (xx - mu[0]), color=GREEN, lw=1.4)
ax.axvline(xa, color=RED, lw=1.3); ax.scatter([xa], [3.0], s=34, color=GREEN, edgecolor='white', zorder=6)
ax.scatter([mu[0]], [mu[1]], s=22, color=BLUE, zorder=6)
ax.text(xa + 0.15, -2.25, '$x_1$ = 3 observed', color=RED, fontsize=9)
ax.text(-3.3, -2.25, 'green: E[$x_2$ | $x_1$]', color=GREEN, fontsize=9)
ax.text(-3.3, 5.45, 'blue: 1 and 2 standard\ndeviations', color=BLUE, fontsize=9, va='top')
ax.set_xlim(-3.5, 6.0); ax.set_ylim(-2.6, 5.6); ax.set_xlabel('$x_1$', fontsize=9.5); ax.set_ylabel('$x_2$', fontsize=9.5); ax.tick_params(labelsize=8.5)
ax = axs[1]; t = np.linspace(-3, 8, 400)
nrm = lambda t, m, v: np.exp(-(t - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
ax.fill_between(t, nrm(t, 2, 3), color=LGREY, ec=GREY, lw=1.0); ax.fill_between(t, nrm(t, 3, 2), color='#BFE3CC', alpha=0.8, ec=GREEN, lw=1.5)
ax.text(-2.9, 0.405, 'marginal of $x_2$:\nmean 2, variance 3', color=GREY, fontsize=9, va='top')
ax.text(8.2, 0.405, 'given $x_1$ = 3:\nmean 3, variance 2', color=GREEN, fontsize=9, ha='right', va='top', weight='bold')
ax.set_xlim(-3, 8.2); ax.set_ylim(0, 0.42); ax.set_yticks([]); ax.set_xlabel('$x_2$', fontsize=9.5); ax.tick_params(labelsize=8.5)
ax.spines['left'].set_visible(False)
fig.tight_layout(w_pad=1.0); save(fig, 'f041.png')

# ---------------------------------------------------------------- R.5-1  KKT: geometry and the dual function
c = np.array([3., 4.]); f = lambda w: np.sum((w - c) ** 2)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.65), dpi=240, gridspec_kw={'width_ratios': [1.05, 1]})
ax = axs[0]; t = np.linspace(0, 2 * np.pi, 400)
for rr in (1, 2, 3, 4, 5): ax.plot(c[0] + rr * np.cos(t), c[1] + rr * np.sin(t), color=(BLUE if rr == 4 else '#9DB8E0'), lw=(1.5 if rr == 4 else 0.8))
ax.fill(np.cos(t), np.sin(t), color='#BFE3CC', alpha=0.9, zorder=3); ax.plot(np.cos(t), np.sin(t), color=GREEN, lw=1.6, zorder=4)
ws = np.array([0.6, 0.8])
d_ = np.array([0.6, 0.8]); arr(ax, tuple(ws), tuple(ws - 1.25 * d_), c=RED, lw=1.5, ms=8, zorder=6)
arr(ax, tuple(ws), tuple(ws + 1.25 * d_), c=GREEN, lw=1.5, ms=8, zorder=6)
ax.scatter(*ws, s=26, color=NAVY, zorder=7); ax.scatter(*c, s=34, color=BLUE, edgecolor='white', zorder=7)
ax.text(c[0] + 0.2, c[1] + 0.2, 'c = (3, 4)', color=BLUE, fontsize=9)
ax.text(1.0, 0.8, '$w^*$', color=NAVY, fontsize=9.5, ha='left', va='bottom'); ax.text(-0.3, 1.45, '$\\nabla f$', color=RED, fontsize=9.5, ha='right')
ax.text(1.5, 1.95, '$\\lambda\\nabla g$', color=GREEN, fontsize=9.5, ha='left')
ax.text(7.5, -0.9, 'feasible: |w| ≤ 1', color=GREEN, fontsize=9, ha='right', bbox=dict(fc='white', ec='none', pad=1.0))
ax.set_aspect('equal'); ax.set_xlim(-1.6, 7.6); ax.set_ylim(-1.5, 7.2); ax.tick_params(labelsize=8.5)
ax.set_xlabel('$w_1$', fontsize=9.5); ax.set_ylabel('$w_2$', fontsize=9.5)
ax = axs[1]; q = lambda l: 25 * l / (1 + l) - l; ll = np.linspace(0, 10, 300)
ax.plot(ll, q(ll), color=PUR, lw=1.8)
ax.axhline(16, color=GREEN, lw=1.2, ls='--'); ax.axhline(25, color=RED, lw=1.0, ls=':')
ax.scatter([4], [16], s=30, color=PUR, zorder=5, edgecolor='white'); ax.scatter([2], [q(2)], s=24, color=PUR, zorder=5, edgecolor='white')
ax.annotate('', xy=(2, 24.6), xytext=(2, q(2) + 0.4), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.0))
ax.text(2.25, 20.5, 'gap\n10.33', color=RED, fontsize=9, va='center')
ax.text(10, 25.7, 'f(0) = 25, a feasible point', color=RED, fontsize=9, ha='right')
ax.text(10, 16.7, 'minimum f* = 16', color=GREEN, fontsize=9, ha='right')
ax.text(4.25, 12.2, 'max q(λ) = 16 at λ = 4', color=PUR, fontsize=9, ha='left'); ax.text(2.0, q(2) - 2.2, 'q(2)', color=PUR, fontsize=9, ha='center')
ax.set_xlim(0, 10); ax.set_ylim(0, 28.5); ax.set_xlabel('multiplier  λ', fontsize=9.5); ax.set_ylabel('dual function  q(λ)', fontsize=9.5); ax.tick_params(labelsize=8.5)
fig.tight_layout(w_pad=1.0); save(fig, 'f051.png')

# ---------------------------------------------------------------- R.6-1  U-curve and the protocol
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.7), dpi=240, gridspec_kw={'width_ratios': [1.0, 1.0]})
ax = axs[0]; c_ = np.linspace(0.3, 5.0, 200)
tr = 0.25 + 2.2 * np.exp(-1.15 * c_); va = 0.55 + 2.0 * np.exp(-1.3 * c_) + 0.055 * np.exp(0.85 * c_) * 0.55
ax.plot(c_, tr, color=BLUE, lw=1.8); ax.plot(c_, va, color=OR, lw=1.8, ls='--')
k = np.argmin(va); ax.axvline(c_[k], color=PUR, lw=1.2, ls=':'); NUM['u_min'] = [round(float(c_[k]), 3), round(float(va[k]), 3)]
ax.text(0.45, 0.35, 'training error', color=BLUE, fontsize=9.2, weight='bold', va='top'); ax.text(3.45, 2.95, 'validation\nerror', color=OR, fontsize=9.2, weight='bold', ha='left', va='top')
ax.text(c_[k] + 0.07, 2.65, 'best\nsetting', color=PUR, fontsize=9.2, va='top')
ax.text(0.45, 2.95, 'high bias', color=GREY, fontsize=9, va='top'); ax.text(4.95, 1.35, 'high\nvariance', color=GREY, fontsize=9, ha='right', va='top')
ax.set_xlim(0.3, 5.0); ax.set_ylim(0, 3.1); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('model complexity (smaller α, larger depth, …)', fontsize=9); ax.set_ylabel('error', fontsize=9.5)
ax = axs[1]; ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10.6)
steps = [('1  Split off a test part', 'fixed seed, stratified', LGREY, GREY), ('2  Cross-validate', 'candidates, training part only', LB, BLUE),
         ('3  Choose the best', 'by the mean fold score', '#EADCF3', PUR), ('4  Refit', 'on the whole training part', '#E1F2E8', GREEN),
         ('5  Score the test part', 'once; report it', LGREY, GREY)]
for j, (t1, t2, fc, ec) in enumerate(steps):
    y0 = 8.9 - j * 2.15
    ax.add_patch(FancyBboxPatch((0.2, y0), 9.6, 1.55, boxstyle='round,pad=0.02,rounding_size=0.2', fc=fc, ec=ec, lw=1.0))
    ax.text(5, y0 + 1.08, t1, ha='center', va='center', fontsize=9.2, weight='bold', color=NAVY); ax.text(5, y0 + 0.42, t2, ha='center', va='center', fontsize=8.3, color=NAVY)
    if j < 4: arr(ax, (5, y0 - 0.02), (5, y0 - 0.58), c=NAVY, lw=1.0, ms=7)
fig.tight_layout(w_pad=1.2); save(fig, 'f061.png')

# ---------------------------------------------------------------- R.7-1  losses of the residual and of the margin
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.65), dpi=240)
ax = axs[0]; r_ = np.linspace(-3, 3, 600); eps = 1.35
hub = np.where(np.abs(r_) <= eps, r_ ** 2, 2 * eps * np.abs(r_) - eps ** 2)
pin = np.where(r_ >= 0, 0.9 * r_, -0.1 * r_)
ax.plot(r_, r_ ** 2, color=NAVY, lw=1.6, label='squared'); ax.plot(r_, np.abs(r_), color=BLUE, lw=1.6, ls='--', label='absolute')
ax.plot(r_, hub, color=GREEN, lw=1.8, label='Huber, ε = 1.35'); ax.plot(r_, pin, color=OR, lw=1.8, ls='-.', label='pinball, τ = 0.9')
ax.set_xlim(-3, 3); ax.set_ylim(-0.2, 6); ax.legend(loc='upper center', fontsize=9, handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
ax.set_xlabel('residual  r = y − ŷ', fontsize=9.5); ax.set_ylabel('loss', fontsize=9.5); ax.tick_params(labelsize=8.5)
ax.set_title('regression', fontsize=9.5, color=NAVY, pad=4)
ax = axs[1]; z = np.linspace(-2.5, 2.5, 600)
ax.plot(z, (z <= 0).astype(float), color=GREY, lw=1.0, label='0–1')
ax.plot(z, np.maximum(0, -z), color=PUR, lw=1.6, ls=':', label='perceptron')
ax.plot(z, np.maximum(0, 1 - z), color=NAVY, lw=1.6, label='hinge')
ax.plot(z, np.maximum(0, 1 - z) ** 2, color=GREEN, lw=1.8, label='squared hinge')
mh = np.where(z >= -1, np.maximum(0, 1 - z) ** 2, -4 * z); ax.plot(z, mh, color=OR, lw=1.4, ls='--', label='modified Huber')
ax.plot(z, np.log1p(np.exp(-z)), color=RED, lw=1.8, label='log loss')
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-0.1, 3.6); ax.legend(loc='upper right', fontsize=8.0, handlelength=1.8, borderaxespad=0.2, labelspacing=0.2)
ax.set_xlabel('margin  z = y f(x)', fontsize=9.5); ax.tick_params(labelsize=8.5)
ax.set_title('classification', fontsize=9.5, color=NAVY, pad=4)
fig.tight_layout(w_pad=1.0); save(fig, 'f071.png')
json.dump(NUM, open('verify/fig_numbers.json', 'w'), indent=1)
print('figs ok', NUM)

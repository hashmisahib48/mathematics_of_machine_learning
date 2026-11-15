# Problem 50C.2: the hour of the day. Writes fig/s50c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import SplineTransformer
from sklearn.tree import DecisionTreeRegressor

# a daily pattern with its peak just before midnight: y = 10 exp(-d^2 / 18) + noise, d the circular distance to 23 h
rng = np.random.RandomState(0); h = np.tile(np.arange(24), 30); day = np.repeat(np.arange(30), 24)
d = np.minimum(np.abs(h - 23), 24 - np.abs(h - 23)); y = 10 * np.exp(-d ** 2 / 18) + rng.normal(0, 0.7, len(h))
tr = day < 20; th = 2 * np.pi * h / 24
enc = {"hour as 0..23": lambda v: v[:, None].astype(float),
       "sin, cos": lambda v: np.column_stack([np.sin(2 * np.pi * v / 24), np.cos(2 * np.pi * v / 24)]),
       "one-hot": lambda v: (v[:, None] == np.arange(24)).astype(float),
       "periodic splines": None}
sp = SplineTransformer(n_knots=7, degree=3, knots=np.linspace(0, 24, 7)[:, None], extrapolation="periodic").fit(np.arange(24.0)[:, None])
enc["periodic splines"] = lambda v: sp.transform(v[:, None].astype(float))
print("circular distance between hours 23 and 0, and 11 and 12 (sin/cos points on the unit circle):")
f = enc["sin, cos"]; dist = lambda a, b: np.linalg.norm(f(np.array([a]))[0] - f(np.array([b]))[0])
print("  raw distances 23 and 1;  sin/cos distances %.4f and %.4f" % (dist(23, 0), dist(11, 12)))
grid = np.arange(24); fits = {}
for k, e in enc.items():
    m = Ridge(1e-3).fit(e(h[tr]), y[tr]); fits[k] = m.predict(e(grid)); print("  ridge on %-18s columns %2d   test R2 %.3f" % (k, e(grid).shape[1], r2_score(y[~tr], m.predict(e(h[~tr])))))
t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(h[tr, None], y[tr]); print("  tree on the raw hour (depth 3)    test R2 %.3f" % r2_score(y[~tr], t.predict(h[~tr, None])))
kn = KNeighborsRegressor(60).fit(enc["hour as 0..23"](h[tr]), y[tr]); print("  60-NN on the raw hour             test R2 %.3f" % r2_score(y[~tr], kn.predict(enc["hour as 0..23"](h[~tr]))))
kn = KNeighborsRegressor(60).fit(enc["sin, cos"](h[tr]), y[tr]); print("  60-NN on sin, cos                 test R2 %.3f" % r2_score(y[~tr], kn.predict(enc["sin, cos"](h[~tr]))))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.8), dpi=240, gridspec_kw={'width_ratios': [1.7, 1]})
ax = axs[0]; ax.plot(h[tr] + rng.uniform(-0.25, 0.25, tr.sum()), y[tr], '.', color=LGREY, ms=2.5, zorder=1)
for (k, v), col, ls in zip(fits.items(), (RED, BLUE, GREEN, OR), ('-', '-', ':', '--')): ax.plot(grid, v, color=col, lw=1.2, ls=ls, label=k)
ax.set_xlabel('hour of the day', fontsize=8.5); ax.set_ylabel('$y$', fontsize=8.5); ax.set_xticks([0, 6, 12, 18, 23]); ax.tick_params(labelsize=8)
ax.legend(fontsize=7.6, loc='upper center', ncol=2, handlelength=1.4, columnspacing=0.8, bbox_to_anchor=(0.5, 1.28)); ax.set_ylim(-1.5, 12)
ax = axs[1]; a = np.linspace(0, 2 * np.pi, 24, endpoint=False); ax.plot(np.sin(a), np.cos(a), 'o', color=BLUE, ms=3)
ax.plot(np.sin(a[[23, 0]]), np.cos(a[[23, 0]]), color=RED, lw=1.6)
for hh in (0, 6, 12, 18): ax.text(1.3 * np.sin(a[hh]), 1.3 * np.cos(a[hh]), str(hh), ha='center', va='center', fontsize=8.5, color=NAVY)
ax.text(0, 0, 'sin, cos\nplane', fontsize=8, ha='center', va='center', color=GREY)
ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s50c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s50c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

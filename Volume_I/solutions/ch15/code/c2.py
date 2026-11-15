# Problem 15C.2: where uniform and quantile knots put the data, and quantile knots with ties.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
import warnings
np.set_printoptions(precision=3, suppress=True)
f = lambda x: 4*np.exp(-(x-3)**2/0.8) + 0.5*x
rng = np.random.default_rng(0)
x = np.sort(rng.exponential(2.5, 80)); x = x[x < 10]
print("the 78 samples of Problem 15B.2: min %.3f  median %.3f  max %.3f" % (x.min(), np.median(x), x.max()))
for kind in ("uniform", "quantile"):
    t = SplineTransformer(n_knots=7, knots=kind).fit(x[:, None]).bsplines_[0].t[3:-3]
    cnt = np.histogram(x, bins=t)[0]
    print(kind, "knots", t, "\n   samples per interval", cnt.tolist(), " widths", np.diff(t))
# ties: 50 per cent of the samples have exactly x = 0 (a sensor that reads zero when idle)
xt = np.r_[np.zeros(60), np.sort(rng.exponential(2.5, 60))]; xt = xt[xt < 10]
yt = f(xt) + rng.normal(0, 0.4, len(xt))
kq = np.quantile(xt, np.linspace(0, 1, 7))
print("\nwith 60 of 120 values equal to 0, the quantile knots are", kq)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sp = SplineTransformer(n_knots=7, knots="quantile", include_bias=False).fit(xt[:, None])
    Phi = sp.transform(xt[:, None]); m = make_pipeline(sp, LinearRegression()).fit(xt[:, None], yt)
print("warnings:", len(w), " columns", Phi.shape[1], " rank of Phi:", np.linalg.matrix_rank(Phi),
      " all-zero columns:", int((np.abs(Phi).sum(0) == 0).sum()))
print("knot vector:", sp.bsplines_[0].t)
print("fitted values at x = 0 and 0.5 and 1:", m.predict([[0.], [0.5], [1.0]]))
ku = SplineTransformer(n_knots=7, knots="uniform", include_bias=False)
mu = make_pipeline(ku, LinearRegression()).fit(xt[:, None], yt)
print("uniform knots on the same data:", mu.predict([[0.], [0.5], [1.0]]), " (curve: ", f(np.array([0., .5, 1.])), ")")
print("rank of Phi with uniform knots:", np.linalg.matrix_rank(ku.fit(xt[:, None]).transform(xt[:, None])))

fig, axs = plt.subplots(1, 2, figsize=(5.4, 1.9), dpi=240)
ax = axs[0]; xx = np.linspace(0, 10, 400)[:, None]
ax.hist(x, bins=20, color=LGREY, edgecolor='white')
for kind, yy, col in (("uniform", -2.2, BLUE), ("quantile", -4.6, GREEN)):
    t = SplineTransformer(n_knots=7, knots=kind).fit(x[:, None]).bsplines_[0].t[3:-3]
    ax.plot(t, np.full(7, yy), '|', color=col, ms=8, mew=1.7, clip_on=False)
    ax.text(10.1, yy, kind, color=col, fontsize=8, va='center')
ax.set_xlabel('$x$', labelpad=14); ax.set_ylabel('samples'); ax.set_xlim(0, 10)
ax = axs[1]
ax.scatter(xt, yt, s=8, color=GRID, zorder=1)
ax.plot(xx, f(xx[:, 0]), color=GREY, lw=0.9, ls=':')
ax.plot(xx, m.predict(xx), color=GREEN, lw=1.5, label='quantile knots')
ax.plot(xx, mu.predict(xx), color=BLUE, lw=1.3, ls='--', label='uniform knots')
ax.set_xlabel('$x$ (half the samples at 0)'); ax.set_ylabel('$y$'); ax.set_ylim(-1.2, 8.2); ax.legend(fontsize=8, loc='upper center', ncol=2, columnspacing=1.0, handlelength=1.8); ax.set_xlim(0, 10)
fig.tight_layout(); fs = min(t_.get_fontsize() for t_ in fig.findobj(matplotlib.text.Text) if t_.get_text().strip()); save(fig, 'fig/s15c2.png'); print("smallest text", fs)

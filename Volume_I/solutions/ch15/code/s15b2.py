# Problem 15B.2: uniform against quantile knots on a skewed feature.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

f = lambda x: 4 * np.exp(-(x - 3) ** 2 / 0.8) + 0.5 * x   # §15.9
rng = np.random.default_rng(0)
x = np.sort(rng.exponential(2.5, 80))
x = x[x < 10][:, None]                                    # training x
y = f(x[:, 0]) + rng.normal(0, 0.4, len(x))
xt = rng.exponential(2.5, 3000)
xt = xt[xt < 10][:, None]                                 # test x
yt = f(xt[:, 0]) + rng.normal(0, 0.4, len(xt))
print("n_train =", len(x), " n_test =", len(xt))

rmse = {"uniform": [], "quantile": []}
ks = list(range(4, 17))
for kind in rmse:
    for k in ks:
        sp = SplineTransformer(n_knots=k, knots=kind,
                               include_bias=False)
        m = make_pipeline(sp, LinearRegression()).fit(x, y)
        e = m.predict(xt) - yt
        rmse[kind].append(np.sqrt(np.mean(e ** 2)))
print("knots  uniform  quantile")
for k, u, q in zip(ks, *rmse.values()):
    print(f"{k:5d} {u:8.3f} {q:9.3f}")
for kind in rmse:
    j = int(np.argmin(rmse[kind]))
    print(f"best {kind:8s}: {ks[j]} knots, RMSE {rmse[kind][j]:.3f}")
for kind in rmse:               # samples per knot interval, 16 knots
    sp = SplineTransformer(n_knots=16, knots=kind).fit(x)
    t = sp.bsplines_[0].t[3:-3]
    cnt = np.histogram(x[:, 0], bins=t)[0]
    print(kind, "at 16 knots, samples per interval:", cnt.tolist())
    poor = np.flatnonzero(cnt <= 2)       # 0, 1 or 2 samples
    hit = np.isin(np.digitize(xt[:, 0], t[1:-1]), poor)
    print(f"   test share in the {len(poor)} intervals with at"
          f" most 2: {hit.mean():.3f}")

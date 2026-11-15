# Leakage of a target encoding fitted and applied on the same rows.
import json, os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

HERE = os.path.dirname(os.path.abspath(__file__))

def noise_table(n=1200, k=400, seed=0):
    rng = np.random.RandomState(seed)
    ids = rng.permutation(np.repeat(np.arange(k), n // k))
    return ids.reshape(-1, 1).astype(str), rng.normal(size=n)   # y unrelated

def run(smooth=5.0):
    X, y = noise_table(); Xt, yt = noise_table(seed=1)
    kf = KFold(5, shuffle=True, random_state=0)
    out = {}
    for name in ("fit.transform", "fit_transform"):
        te = TargetEncoder(smooth=smooth, target_type="continuous", cv=kf)
        if name == "fit.transform":
            Z = te.fit(X, y).transform(X)
        else:
            Z = te.fit_transform(X, y)
        m = LinearRegression().fit(Z, y)
        out[name] = dict(corr=float(np.corrcoef(Z[:, 0], y)[0, 1]),
                         train_r2=float(m.score(Z, y)),
                         test_r2=float(m.score(te.transform(Xt), yt)),
                         Z=Z[:, 0].tolist())
    return X, y, out

if __name__ == "__main__":
    X, y, out = run()
    for k, v in out.items():
        print(f"{k:14s} corr(Z,y) {v['corr']:.3f}  train R2 "
              f"{v['train_r2']:.3f}  test R2 {v['test_r2']:.3f}")
    # one row alone in its category: weight of its own target
    m = 5.0
    y2 = y.copy(); y2[0] += 1.0
    xs = np.array([["a"], ["b"], ["b"]]); ys = np.array([1., 2, 3])
    f = lambda v: TargetEncoder(smooth=m, target_type="continuous"
                                ).fit(xs, v).transform(xs)[0, 0]
    d = f(ys + np.array([1, 0, 0])) - f(ys)
    print("d t_a / d y_a =", round(d, 4), " formula",
          round(1 / (1 + m) + m / (1 + m) / 3, 4))
    json.dump({k: {a: b for a, b in v.items() if a != "Z"}
               for k, v in out.items()},
              open(os.path.join(HERE, "..", "verify", "leak.json"), "w"))

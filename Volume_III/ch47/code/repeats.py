# Twenty repetitions of the waiting-time experiment of Section 47.9.
import json
import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


def waits(n, seed):
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    t = np.where(short, rng.normal(54, 5, n), rng.normal(80, 6, n))
    return t[:, None]


def truth(x):
    return 0.35 * norm.pdf(x, 54, 5) + 0.65 * norm.pdf(x, 80, 6)


Xte = waits(4000, 999)
res = {k: [] for k in ["true density", "mixture K=2", "KDE, h by CV",
                       "KDE, Scott", "KDE, Silverman"]}
hs = []
for seed in range(20):
    X = waits(200, 100 + seed)
    sd = X.std()
    res["true density"].append(np.log(truth(Xte)).mean())
    gm = GaussianMixture(2, random_state=0).fit(X)
    res["mixture K=2"].append(gm.score(Xte))
    grid = {"bandwidth": np.arange(1.0, 10.01, 0.25)}
    cv = KFold(10, shuffle=True, random_state=0)
    gs = GridSearchCV(KernelDensity(), grid, cv=cv).fit(X)
    hs.append(gs.best_params_["bandwidth"])
    res["KDE, h by CV"].append(gs.best_estimator_.score(Xte) / len(Xte))
    for rule, key in [("scott", "KDE, Scott"),
                      ("silverman", "KDE, Silverman")]:
        h = KernelDensity(bandwidth=rule).fit(X).bandwidth_ * sd
        k = KernelDensity(bandwidth=h).fit(X)
        res[key].append(k.score(Xte) / len(Xte))
out = {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
       for k, v in res.items()}
out["h_cv_mean"] = round(float(np.mean(hs)), 2)
out["h_cv_min_max"] = [float(min(hs)), float(max(hs))]
json.dump(out, open("verify/repeats.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:16s}", v)

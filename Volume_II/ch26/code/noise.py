# Label noise: 10% of the training labels flipped, 10 repetitions.
import json
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

R, M, N = 10, 400, 300
AT = (1, 10, 50, 100, 400)
curves = {}                     # (depth, clean or noisy) -> errors
share = {1: [], 3: []}          # weight on the flipped points
for rep in range(R):
    X, y = make_moons(N + 1000, noise=0.3, random_state=rep)
    Xtr, ytr, Xte, yte = X[:N], y[:N], X[N:], y[N:]
    rng = np.random.default_rng(rep)
    flip = rng.random(N) < 0.10
    for d in (1, 3):
        noisy = np.where(flip, 1 - ytr, ytr)
        for lab, yy in (("clean", ytr), ("noisy", noisy)):
            m = AdaBoostClassifier(DTC(max_depth=d), n_estimators=M,
                                   random_state=0).fit(Xtr, yy)
            sc = list(m.staged_score(Xte, yte))
            err = 1 - np.array(sc)
            curves.setdefault((d, lab), []).append(err)
            if lab == "noisy":
                w, s = np.full(N, 1 / N), []
                for e, a in zip(m.estimators_,
                                m.estimator_weights_):
                    s.append(w[flip].sum() / w.sum())
                    w = w * np.exp(a * (e.predict(Xtr) != yy))
                    w /= w.sum()
                share[d].append(s)
out = {f"{d}_{lab}": np.mean(v, axis=0).tolist()
       for (d, lab), v in curves.items()}
out.update({f"share_{d}": np.mean(v, axis=0).tolist()
            for d, v in share.items()})
json.dump(out, open("../verify/noise.json", "w"))
print("mean test error over", R, "repetitions;")
print("share = weight carried by the flipped points")
print("rounds          1     10     50    100    400")
for d in (1, 3):
    for lab in ("clean", "noisy"):
        e = out[f"{d}_{lab}"]
        row = " ".join(f"{e[k-1]:6.3f}" for k in AT)
        print(f"depth {d} {lab:5s} " + row)
    s = out[f"share_{d}"]
    row = " ".join(f"{s[k-1]:6.3f}" for k in AT)
    print(f"depth {d} share " + row)

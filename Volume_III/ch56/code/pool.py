# Sections 56.2, 56.8: a pool of 300 random candidates, scored once,
# and square grids of 9 to 64 cells (writes verify/pool.json).
import json
import numpy as np
from scipy.stats import loguniform
from sklearn.model_selection import GridSearchCV, ParameterSampler
from common import cancer_split, svc_pipe, GRID

Xtr, Xte, ytr, yte = cancer_split()
lo = dict(C=(-5, 9), g=(-15, 1))            # log2 ranges of Chapter 22
dist = {"svc__C": loguniform(2.0 ** lo["C"][0], 2.0 ** lo["C"][1]),
        "svc__gamma": loguniform(2.0 ** lo["g"][0], 2.0 ** lo["g"][1])}
pool = list(ParameterSampler(dist, 300, random_state=0))
gs = GridSearchCV(svc_pipe(), [{k: [v] for k, v in p.items()}
                               for p in pool], cv=5).fit(Xtr, ytr)
cv = gs.cv_results_["mean_test_score"]
test = np.array([svc_pipe().set_params(**p).fit(Xtr, ytr)
                 .score(Xte, yte) for p in pool])
out = {"C": [float(p["svc__C"]) for p in pool],
       "gamma": [float(p["svc__gamma"]) for p in pool],
       "cv": cv.tolist(), "test": test.tolist(), "grids": {}}
print("pool: best CV", round(cv.max(), 4), " test of that one",
      round(test[cv.argmax()], 4), " mean test", round(test.mean(), 4))
for m in range(3, 9):
    g = {"svc__C": 2.0 ** np.linspace(*lo["C"], m),
         "svc__gamma": 2.0 ** np.linspace(*lo["g"], m)}
    s = GridSearchCV(svc_pipe(), g, cv=5).fit(Xtr, ytr)
    out["grids"][m * m] = [float(s.best_score_),
                           float(s.score(Xte, yte))]
    print(f"grid {m}x{m}: CV {s.best_score_:.4f}"
          f" test {s.score(Xte, yte):.4f}")
# an inert hyper-parameter: degree is ignored by the RBF kernel
g2 = {"svc__C": [1, 100], "svc__gamma": [0.001, 0.01],
      "svc__degree": [2, 3, 4]}
r = GridSearchCV(svc_pipe(), g2, cv=5).fit(Xtr, ytr)
m = r.cv_results_["mean_test_score"].reshape(2, 3, 2)  # C, degree, g
print("degree changes the score by", float(np.ptp(m, axis=1).max()))
json.dump(out, open("verify/pool.json", "w"))

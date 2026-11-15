# Section 56.9: three searches and a nested estimate on the cancer data.
import json
import numpy as np
from scipy.stats import loguniform
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (GridSearchCV, HalvingGridSearchCV,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score)
from common import cancer_split, svc_pipe, GRID

Xtr, Xte, ytr, yte = cancer_split()
dist = {"svc__C": loguniform(2.0**-5, 2.0**9),
        "svc__gamma": loguniform(2.0**-15, 2.0)}
searches = {
    "grid": GridSearchCV(svc_pipe(), GRID, cv=5),
    "random": RandomizedSearchCV(svc_pipe(), dist, n_iter=72, cv=5,
                                 random_state=0),
    "halving": HalvingGridSearchCV(svc_pipe(), GRID, factor=2, cv=5,
                                   random_state=0)}
print("method   evals  fits   cost   CV      test"
      "    log2 C  log2 g")
cost = {}
for name, s in searches.items():
    s.fit(Xtr, ytr)
    if name == "halving":
        evals = sum(s.n_candidates_)
        cost[name] = sum(a * b for a, b in zip(
            s.n_candidates_, s.n_resources_)) / 426
    else:
        evals, cost[name] = len(s.cv_results_["params"]), 72.0
    p = s.best_params_
    print(f"{name:8s}{evals:6d}{evals * 5 + 1:6d}{cost[name]:7.1f}"
          f"  {s.best_score_:.4f}  {s.score(Xte, yte):.4f}"
          f"{np.log2(p['svc__C']):8.2f}"
          f"{np.log2(p['svc__gamma']):8.2f}")
dcv = cross_val_score(svc_pipe(), Xtr, ytr, cv=5).mean()
print("defaults: CV", round(dcv, 4), " test",
      round(svc_pipe().fit(Xtr, ytr).score(Xte, yte), 4))
outer = StratifiedKFold(5, shuffle=True, random_state=0)
nest = cross_val_score(GridSearchCV(svc_pipe(), GRID, cv=5),
                       Xtr, ytr, cv=outer)
print("nested folds", np.round(nest, 4),
      " mean", round(nest.mean(), 4))
# (the lines below save numbers and results for the figures)
res = {}
for name, s in searches.items():
    p = s.best_params_
    res[name] = dict(evals=len(s.cv_results_["params"]),
                     fits=len(s.cv_results_["params"]) * 5 + 1,
                     rows=cost[name], cv=s.best_score_,
                     test=s.score(Xte, yte),
                     c=float(np.log2(p["svc__C"])),
                     g=float(np.log2(p["svc__gamma"])))
res["nested"] = nest.tolist()
h = searches["halving"].cv_results_
res["halving_path"] = [[int(i), int(n), float(m), float(p["svc__C"]),
                        float(p["svc__gamma"])] for i, n, m, p in zip(
    h["iter"], h["n_resources"], h["mean_test_score"], h["params"])]
res["grid_scores"] = searches["grid"].cv_results_[
    "mean_test_score"].tolist()
rcv = searches["random"].cv_results_
res["random_points"] = [[float(p["svc__C"]), float(p["svc__gamma"]),
                         float(m)] for p, m in zip(
    rcv["params"], rcv["mean_test_score"])]
json.dump(res, open("verify/case_search.json", "w"))

# Section 56.7: random search and successive halving.
from scipy.stats import loguniform
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import (HalvingGridSearchCV,
                                     RandomizedSearchCV)
from common import cancer_split, svc_pipe, GRID

Xtr, Xte, ytr, yte = cancer_split()
dist = {"svc__C": loguniform(2**-5, 2**9),
        "svc__gamma": loguniform(2**-15, 2)}
rs = RandomizedSearchCV(svc_pipe(), dist, n_iter=72, cv=5,
                        random_state=0).fit(Xtr, ytr)
hs = HalvingGridSearchCV(svc_pipe(), GRID, factor=2, cv=5,
                         random_state=0).fit(Xtr, ytr)
for name, s in (("random", rs), ("halving", hs)):
    b = s.best_params_
    print(f"{name:8s} CV {s.best_score_:.4f}  test"
          f" {s.score(Xte, yte):.4f}  C {b['svc__C']:.3g}"
          f"  gamma {b['svc__gamma']:.3g}")
print("halving rows ", hs.n_resources_)
print("halving cands", hs.n_candidates_)
print("iterations", hs.n_iterations_, " required",
      hs.n_required_iterations_, " possible",
      hs.n_possible_iterations_)
print("rows in cv_results_:", len(hs.cv_results_["params"]),
      " best rung:", hs.cv_results_["iter"][hs.best_index_])

# Numbers quoted in the prose of Chapter 56.
import json, math
import numpy as np
from scipy.stats import loguniform, uniform

print("uniform on (1e-5, 1): P(x > 0.1) =",
      round(float(uniform(1e-5, 1 - 1e-5).sf(0.1)), 3))
x = loguniform(1e-5, 1).rvs(100000, random_state=0)
print("loguniform: share above 0.1:", round((x > 0.1).mean(), 3),
      " per decade 0.2")
rng = np.random.RandomState(0)
for M in (1, 4, 16, 72, 300):
    mx = rng.randn(20000, M).max(axis=1).mean()
    print(f"M={M:3d}: E max of M normals {mx:.3f}  bound"
          f" {math.sqrt(2 * math.log(M)):.3f}")
print("halving: 81 x 2430 =", 81 * 2430, " rungs 5 x 2430 =", 5 * 2430,
      " ratio", round(81 * 2430 / (5 * 2430), 1))
print("grid sizes, 5 values each: p=3", 5**3, " p=5", 5**5, " p=7",
      5**7, " fits with K=5:", 5 * 5**5)
print("nested fits: 5 x (72 x 5 + 1) =", 5 * (72 * 5 + 1))
print("one patient in 426 =", round(1 / 426, 5),
      " 0.9906 -> correct", round(0.9906 * 426, 1),
      " 0.9883 ->", round(0.9883 * 426, 1))
d = json.load(open("verify/pool.json"))
cv = np.array(d["cv"]); n = len(cv)
for thr in (0.98, 0.985):
    q = (cv >= thr).mean()
    n95 = math.ceil(math.log(0.05) / math.log(1 - q))
    perm = np.array([rng.permutation(n) for _ in range(2000)])
    first = np.array([np.argmax(cv[p] >= thr) + 1 if (cv[p] >= thr).any()
                      else n + 1 for p in perm])
    print(f"threshold {thr}: q = {q:.3f}, Eq. 56.3-3 gives n = {n95},"
          f" observed 95th percentile {np.percentile(first, 95):.0f},"
          f" median {np.median(first):.0f}")

W = np.array(json.load(open("verify/winner.json")))
print("12 re-splits: means", np.round(W.mean(axis=0), 4),
      " sd of fresh", round(W[:, 2].std(ddof=1), 4),
      " sd of inner CV", round(W[:, 0].std(ddof=1), 4))
print("best_score_ above fresh in", int((W[:, 0] > W[:, 2]).sum()),
      "of 12; nested above fresh in", int((W[:, 1] > W[:, 2]).sum()))
print("gaps: inner-fresh", round((W[:, 0] - W[:, 2]).mean(), 4),
      " nested-fresh", round((W[:, 1] - W[:, 2]).mean(), 4))

# a float-rounding trap in the halving schedule: 243 = 3**5
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
print("math.log(81, 3) =", math.log(81, 3),
      " math.log(243, 3) =", math.log(243, 3))
g = {"max_depth": [2, 4, 8], "min_samples_leaf": [1, 5, 20],
     "ccp_alpha": [0.0, 0.002, 0.01], "max_features": [.3, .6, 1.],
     "min_samples_split": [2, 10, 40]}
X, y = make_classification(n_samples=2430, n_informative=6,
                           random_state=0)
h = HalvingGridSearchCV(DecisionTreeClassifier(random_state=0), g,
                        factor=3, random_state=0).fit(X, y)
print("243 candidates: required", h.n_required_iterations_,
      " candidates per rung", h.n_candidates_)

# the ridge of good cells: log2 C + log2 gamma near -4
cs = json.load(open("verify/case_search.json"))
sc = np.array(cs["grid_scores"]).reshape(8, 9)      # rows C, cols gamma
lc, lg = np.arange(-5, 10, 2), np.arange(-15, 2, 2)
top = np.argsort(-sc.ravel(), kind="stable")[:10]
sums = sorted(int(lc[i // 9] + lg[i % 9]) for i in top)
print("top-10 cells: log2 C + log2 gamma =", sums,
      " scores", np.round(sc.ravel()[top], 4).tolist())
print("cells with accuracy >= 0.98:", int((sc >= 0.98).sum()),
      " of 72;  below 0.90:", int((sc < 0.90).sum()))

from sklearn.model_selection import GridSearchCV
from common import cancer_split, svc_pipe, GRID
Xtr, Xte, ytr, yte = cancer_split()
one = GridSearchCV(svc_pipe(), GRID, cv=5).fit(Xtr, ytr)
print("keys of cv_results_ with one scorer, no train scores:",
      len(one.cv_results_))
print("standard error of an accuracy of 0.95 on 143 rows:",
      round(math.sqrt(0.95 * 0.05 / 143), 4))

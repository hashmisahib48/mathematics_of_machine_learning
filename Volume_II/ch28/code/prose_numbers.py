# Numbers quoted in the prose of Chapter 28 that no listing prints.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
import numpy as np
from sklearn.datasets import make_friedman1, make_classification, load_iris
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble._hist_gradient_boosting.binning import (
    _find_binning_thresholds)
OUT = {}

# 1. One HGB tree equals the tree of Chapter 24 (squared error, lam 0).
X, y = make_friedman1(200, n_features=5, noise=1.0, random_state=0)
for L in (2, 4, 8):
    h = HGBR(max_iter=1, learning_rate=1.0, max_leaf_nodes=L,
             min_samples_leaf=1, early_stopping=False).fit(X, y)
    d = DecisionTreeRegressor(max_leaf_nodes=L, random_state=0).fit(X, y)
    OUT["tree_diff_%d" % L] = float(np.abs(h.predict(X)
                                           - d.predict(X)).max())
nodes = HGBR(max_iter=1, learning_rate=1.0, max_leaf_nodes=2,
             min_samples_leaf=1, early_stopping=False).fit(
                 X, y)._predictors[0][0].nodes
st = DecisionTreeRegressor(max_depth=1, random_state=0).fit(X, y).tree_
sse_drop = st.impurity[0] * 200 - (st.impurity[1] * st.n_node_samples[1]
                                   + st.impurity[2] * st.n_node_samples[2])
OUT["gain_stump"] = float(nodes["gain"][0])
OUT["sse_decrease_stump"] = float(sse_drop)

# 2. Binning: ties reduce the number of bins.
rng = np.random.default_rng(0)
col = np.where(rng.random(1000) < 0.6, 0.0, rng.normal(size=1000))
OUT["bins_with_60pct_zeros"] = int(len(_find_binning_thresholds(col, 255)) + 1)
col2 = rng.normal(size=1000)
OUT["bins_continuous_1000"] = int(len(_find_binning_thresholds(col2, 255)) + 1)
col3 = rng.integers(0, 30, 1000).astype(float)
OUT["bins_30_levels"] = int(len(_find_binning_thresholds(col3, 255)) + 1)

# 3. NaN at predict time when training had none: goes to the larger child.
Xn = np.arange(40, dtype=float).reshape(-1, 1)
yn = (Xn[:, 0] >= 30).astype(int)           # 30 left, 10 right
m = HGBC(max_iter=1, max_leaf_nodes=2, min_samples_leaf=1,
         early_stopping=False).fit(Xn, yn)
nd = m._predictors[0][0].nodes
OUT["nan_rule"] = {"threshold": float(nd["num_threshold"][0]),
                   "missing_go_to_left": int(nd["missing_go_to_left"][0]),
                   "p_nan": float(m.predict_proba([[np.nan]])[0, 1]),
                   "p_left_leaf": float(m.predict_proba([[5.0]])[0, 1])}

# 4. min_samples_leaf: a node with fewer than 2 x leaf samples is a leaf.
for n in (39, 40):
    Xs, ys = make_friedman1(n, n_features=5, random_state=0)
    mm = HGBR(max_iter=1, min_samples_leaf=20,
              early_stopping=False).fit(Xs, ys)
    OUT["leaves_n%d_msl20" % n] = mm._predictors[0][0].get_n_leaf_nodes()

# 5. Multiclass: one tree per class per iteration.
Xi, yi = load_iris(return_X_y=True)
mi = HGBC(max_iter=5, min_samples_leaf=5, early_stopping=False).fit(Xi, yi)
OUT["iris_trees_per_iter"] = int(mi.n_trees_per_iteration_)
OUT["iris_total_trees"] = int(sum(len(p) for p in mi._predictors))
OUT["has_feature_importances"] = hasattr(mi, "feature_importances_")

# 6. Sum of gradients at the root is zero for the starting score.
p0 = yn.mean()
OUT["root_sum_g"] = float((p0 - yn).sum())

# 7. The by-hand example with lam = 1 (toy data of Listing 28.1).
x = np.array([8, 15, 22, 31, 40, 47, 58, 66, 73, 85, 92, 110.])
yt = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
b = np.searchsorted([26.5, 52.5, 79.0], x)
p = yt.mean(); g = p - yt; hh = np.full(12, p * (1 - p))
G, H = np.bincount(b, g, 4), np.bincount(b, hh, 4)
for lam in (0.0, 1.0):
    GL, HL = np.cumsum(G)[:-1], np.cumsum(H)[:-1]
    GR, HR = G.sum() - GL, H.sum() - HL
    gain = GL**2/(HL+lam) + GR**2/(HR+lam)
    OUT["toy_lam%d" % lam] = {"gain": gain.round(4).tolist(),
        "vL": float(-GL[0]/(HL[0]+lam)), "vR": float(-GR[0]/(HR[0]+lam)),
        "vL1": float(-GL[1]/(HL[1]+lam)), "vR1": float(-GR[1]/(HR[1]+lam))}
mt = HGBC(max_iter=1, max_leaf_nodes=2, min_samples_leaf=1, max_bins=4,
          l2_regularization=1.0, learning_rate=1.0,
          early_stopping=False).fit(x.reshape(-1, 1), yt)
nt = mt._predictors[0][0].nodes
OUT["toy_lam1_sklearn"] = {"gain": float(nt["gain"][0]),
                           "vL": float(nt["value"][1]),
                           "vR": float(nt["value"][2])}
# subtraction trick on the toy histogram
left = np.bincount(b[b <= 0], g[b <= 0], 4)
OUT["toy_right_by_subtraction_G"] = (G - left).round(4).tolist()
OUT["toy_right_direct_G"] = np.bincount(b[b > 0], g[b > 0], 4).round(4).tolist()

# 8. Round 2 of the toy data: total G, H and the three-term gain; and
#    the stored gain of a node below the root (three-term form).
F1 = np.full(12, np.log(p / (1 - p))) + 0.5 * np.where(b <= 0, -1.7142857, 0.5714286)
q = 1 / (1 + np.exp(-F1)); g2, h2 = q - yt, q * (1 - q)
G2, H2 = np.bincount(b, g2, 4), np.bincount(b, h2, 4)
two = G2[0]**2/H2[0] + (G2.sum()-G2[0])**2/(H2.sum()-H2[0])
OUT["round2"] = {"sum_G": float(G2.sum()), "sum_H": float(H2.sum()),
                 "two_term": float(two),
                 "three_term": float(two - G2.sum()**2/H2.sum())}
m3 = HGBC(max_iter=1, max_leaf_nodes=3, min_samples_leaf=1, max_bins=4,
          learning_rate=1.0, early_stopping=False).fit(x.reshape(-1, 1), yt)
n3 = m3._predictors[0][0].nodes
j = [k for k in (1, 2) if not n3["is_leaf"][k]][0]
inn = (b > 0) if j == 2 else (b <= 0)
Gn, Hn = np.bincount(b[inn], g[inn], 4), np.bincount(b[inn], hh[inn], 4)
cl, ch_ = np.cumsum(Gn), np.cumsum(Hn)
bk = int(n3["bin_threshold"][j])
GLn, HLn = cl[bk], ch_[bk]; GRn, HRn = Gn.sum()-GLn, Hn.sum()-HLn
t2 = GLn**2/HLn + GRn**2/HRn
OUT["below_root"] = {"stored": float(n3["gain"][j]), "two_term": float(t2),
                     "three_term": float(t2 - Gn.sum()**2/Hn.sum())}
for k, v in OUT.items():
    print(k, v)
with open("verify/prose_numbers.json", "w") as f:
    json.dump(OUT, f, indent=1)

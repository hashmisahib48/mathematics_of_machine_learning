# Numbers quoted in the prose of Chapter 55.
import json, math, numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     TimeSeriesSplit)
R = {}

# 55.3 split counts
R["lpo_426_2"] = math.comb(426, 2)
R["lpo_100_3"] = math.comb(100, 3)
R["lpo_569_2"] = math.comb(569, 2)
print("C(426,2) =", R["lpo_426_2"], " C(100,3) =", R["lpo_100_3"])

# 55.3 patients model: y_ij = a_i + e_ij, sd(a)=2, sd(e)=1, m=8 visits
rng = np.random.RandomState(0)
n_pat, m = 20000, 8
a = rng.normal(0, 2, (n_pat, 1)); y = a + rng.normal(0, 1, (n_pat, m))
loo_visit = np.empty_like(y)
for j in range(m):
    loo_visit[:, j] = (y.sum(1) - y[:, j]) / (m - 1)   # same patient
mse_same = np.mean((y - loo_visit) ** 2)
a2 = rng.normal(0, 2, (n_pat, 1)); y2 = a2 + rng.normal(0, 1, (n_pat, m))
mse_new = np.mean((y2 - y.mean()) ** 2)               # grand mean
R["mse_same"], R["mse_new"] = round(mse_same, 3), round(mse_new, 3)
print("MSE same patient", R["mse_same"], "theory", round(1 + 1 / 7, 3),
      "| new patient", R["mse_new"], "theory 5")

# 55.3 variance of the CV mean: 200 data sets of 60 samples
Xp, yp = make_classification(20000, n_features=40, n_informative=8,
    n_redundant=4, class_sep=0.8, flip_y=0.05, random_state=0)
lr = LogisticRegression(max_iter=500)
rng = np.random.RandomState(1)
scores = []
for d in range(200):
    idx = rng.choice(10000, 60, replace=False)
    f = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(idx):
        f.append(lr.fit(Xp[idx[tr]], yp[idx[tr]]).score(
            Xp[idx[te]], yp[idx[te]]))
    scores.append(f)
S = np.array(scores)                                   # (200, 5)
sd_true = S.mean(1).std(ddof=1)
naive = (S.std(1, ddof=1) / np.sqrt(5)).mean()
C = np.corrcoef(S.T); rho = (C.sum() - 5) / 20
R.update(sd_true=round(sd_true, 4), naive_se=round(naive, 4),
         rho=round(rho, 3))
var_cell = S.var(ddof=1)
R["pred_ratio"] = round(float(np.sqrt((1 + 4 * rho) / (1 - rho))), 3)
R["var_cell"] = round(float(var_cell), 5)
print("sd of the 5-fold mean over data sets", R["sd_true"],
      "| mean naive s/sqrt(5)", R["naive_se"], "| mean fold-score corr",
      R["rho"])

# (ratio of the true to the naive standard error, Eq. 55.3-4)
# 55.3 stratified allocation depends on the order of appearance
for lab, yy in (("0 first", [0] * 6 + [1] * 6), ("1 first", [1] * 6 + [0] * 6)):
    cnt = [np.bincount(np.array(yy)[te], minlength=2).tolist() for _, te in
           StratifiedKFold(5).split(np.zeros((12, 1)), yy)]
    print("labels", lab, "per-fold class counts", cnt)
    R["alloc_" + lab] = cnt

# 55.9 sizes in the time-series case
ts = TimeSeriesSplit(5, test_size=30)
R["ts_train_sizes"] = [len(a) for a, _ in ts.split(np.zeros((240, 1)))]
print("train sizes", R["ts_train_sizes"])
json.dump(R, open("verify/prose_numbers.json", "w"))

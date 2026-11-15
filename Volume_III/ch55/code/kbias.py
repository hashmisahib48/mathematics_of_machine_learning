# Bias and spread of the K-fold estimate (invented data, 150 samples).
import json, time, numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

N, NDS = 60, 100
lr = LogisticRegression(C=1.0, max_iter=500)
Ks = [2, 3, 5, 10, 20]
# one population of 20 000; data sets of N rows are drawn from its
# first half, and the second half scores every fitted model
Xp, yp = make_classification(20000, n_features=40, n_informative=8,
    n_redundant=4, class_sep=0.8, flip_y=0.05, random_state=0)
Xt, yt = Xp[10000:], yp[10000:]
rng = np.random.RandomState(1)
t0 = time.time()
est = {K: [] for K in Ks}
seedsd = []
truth = []
for d in range(NDS):
    idx = rng.choice(10000, N, replace=False)
    X, y = Xp[idx], yp[idx]
    truth.append(lr.fit(X, y).score(Xt, yt))
    for K in Ks:
        cv = KFold(K, shuffle=True, random_state=0)
        est[K].append(cross_val_score(lr, X, y, cv=cv).mean())
    if d < 15:               # partition noise on one data set
        seedsd.append([np.std([cross_val_score(lr, X, y,
            cv=KFold(5, shuffle=True, random_state=s)).mean()
            for s in range(20)], ddof=1)])
out = {"truth": float(np.mean(truth)), "Ks": Ks,
       "bias": [float(np.mean(est[K]) - np.mean(truth)) for K in Ks],
       "sd_data": [float(np.std(est[K], ddof=1)) for K in Ks],
       "mean_est": [float(np.mean(est[K])) for K in Ks],
       "sd_partition_5fold": float(np.mean(seedsd))}
print("expected accuracy of a model fitted to", N, "samples:",
      round(out["truth"], 4))
for K, b, s, m in zip(Ks, out["bias"], out["sd_data"], out["mean_est"]):
    print(f"K={K:3d} mean {m:.4f} bias {b:+.4f} sd over data sets {s:.4f}")
print("5-fold, sd over 20 partitions of one data set:",
      round(out["sd_partition_5fold"], 4))
json.dump(out, open("verify/kbias.json", "w"))
print("seconds", round(time.time() - t0, 1))

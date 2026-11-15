# Case study, repeated over 20 random choices of the anomalies.
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
benign = Xtr[ytr == 1]
mal_tr, mal_te = np.where(ytr == 0)[0], np.where(yte == 0)[0]

def make(nov):
    return {"IsolationForest": IsolationForest(random_state=0),
            "LocalOutlierFactor": LocalOutlierFactor(novelty=nov),
            "OneClassSVM": OneClassSVM(nu=0.05),
            "EllipticEnvelope": EllipticEnvelope(
                contamination=0.05, random_state=0)}

def strange(d, A):                      # high = strange; training rows
    if isinstance(d, LocalOutlierFactor):
        return -d.negative_outlier_factor_
    return -d.score_samples(A)

R = {k: {"nov_auc": [], "nov_ap": [], "out_auc": []} for k in make(1)}
roc = {}
for seed in range(20):
    rng = np.random.RandomState(seed)
    T = np.vstack([Xte[yte == 1], Xte[rng.choice(mal_te, 18, False)]])
    t = np.r_[np.zeros(90), np.ones(18)]
    sc = StandardScaler().fit(benign)
    for k, d in make(True).items():     # novelty: train on benign only
        s = -d.fit(sc.transform(benign)).score_samples(sc.transform(T))
        R[k]["nov_auc"].append(roc_auc_score(t, s))
        R[k]["nov_ap"].append(average_precision_score(t, s))
        if seed == 0:
            f, tp, _ = roc_curve(t, s)
            roc[k] = [f.tolist(), tp.tolist()]
    A = np.vstack([benign, Xtr[rng.choice(mal_tr, 14, False)]])
    a = np.r_[np.zeros(len(benign)), np.ones(14)]
    A = StandardScaler().fit_transform(A)   # outlier mode: unclean data
    for k, d in make(False).items():
        R[k]["out_auc"].append(roc_auc_score(a, strange(d.fit(A), A)))
print(f"{'':20s} novelty AUC      novelty AP       outlier AUC")
for k, v in R.items():
    print(f"{k:20s}" + "".join(
        f"  {np.mean(v[m]):.3f} +- {np.std(v[m]):.3f}"
        for m in ("nov_auc", "nov_ap", "out_auc")))
json.dump({"R": R, "roc": roc}, open("verify/case_repeats.json", "w"))

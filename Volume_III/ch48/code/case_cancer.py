# Case study: benign tissue is normal, malignant tissue the novelty.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, average_precision_score

X, y = load_breast_cancer(return_X_y=True)       # y = 1: benign
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
benign = Xtr[ytr == 1]                       # clean training set
rng = np.random.RandomState(0)
mal = rng.choice(np.where(yte == 0)[0], 18, replace=False)
test = np.vstack([Xte[yte == 1], Xte[mal]])   # 90 benign, 18 not
truth = np.r_[np.zeros(90), np.ones(18)]
print("train", benign.shape, " test", test.shape)

def run(scale):                              # fit on benign only
    A, T = benign, test
    if scale:
        sc = StandardScaler().fit(A)
        A, T = sc.transform(A), sc.transform(T)
    out = {}
    for name, d in {
        "IsolationForest": IsolationForest(random_state=0),
        "LocalOutlierFactor": LocalOutlierFactor(novelty=True),
        "OneClassSVM": OneClassSVM(nu=0.05),
        "EllipticEnvelope": EllipticEnvelope(
            contamination=0.05, random_state=0)}.items():
        d.fit(A)
        s, lab = -d.score_samples(T), d.predict(T)
        out[name] = (roc_auc_score(truth, s),
                     average_precision_score(truth, s),
                     (lab[truth == 1] == -1).mean(),
                     (lab[truth == 0] == -1).mean())
    return out
raw, scaled = run(False), run(True)
print(f"{'':20s} raw AUC  AUC    AP  caught  false alarms")
for name, (auc, ap, hit, fa) in scaled.items():
    print(f"{name:20s} {raw[name][0]:7.3f} {auc:5.3f} {ap:5.3f} "
          f"{hit:7.2f} {fa:8.2f}")

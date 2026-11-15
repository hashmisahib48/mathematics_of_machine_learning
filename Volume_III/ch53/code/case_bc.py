# Section 53.11: accuracy against the number of features, honestly.
import json
import numpy as np
from sklearn.feature_selection import (RFE, RFECV, SelectFromModel,
                                       SelectKBest)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bcdata import load

Xa, Xb, ya, yb, names = load()
lr = lambda: LogisticRegression(max_iter=1000)

def score(selector):          # selection is refitted in every fold
    pipe = make_pipeline(StandardScaler(), selector, lr())
    cv = cross_val_score(pipe, Xa, ya, cv=5).mean()
    return cv, pipe.fit(Xa, ya).score(Xb, yb), pipe

ks = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30]
res = {"k": ks, "filter": [], "rfe": [], "l1": []}
for k in ks:
    res["filter"].append(score(SelectKBest(k=k))[:2])
    res["rfe"].append(score(RFE(lr(), n_features_to_select=k))[:2])
for C in (0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0):
    l1 = LogisticRegression(l1_ratio=1, C=C, solver="liblinear",
                            random_state=0)
    cv, te, pipe = score(SelectFromModel(l1))
    n = int(pipe[1].get_support().sum())
    res["l1"].append((C, n, cv, te))
cv, te, pipe = score(RFECV(lr(), cv=5))
res["rfecv"] = [cv, te, int(pipe[1].n_features_)]
for i, k in enumerate(ks):
    if k in (1, 3, 5, 10, 15, 20, 30):
        print("%2d  filter %.4f %.4f   RFE %.4f %.4f" % (
            k, *res["filter"][i], *res["rfe"][i]))
for C, n, cv, te in res["l1"]:
    if C in (0.02, 0.05, 0.2, 1.0):
        print("L1 C=%.2f keeps %2d  %.4f %.4f" % (C, n, cv, te))
print("RFECV keeps %d: %.4f %.4f" % (
    res["rfecv"][2], res["rfecv"][0], res["rfecv"][1]))
json.dump(res, open("verify/case_bc.json", "w"))

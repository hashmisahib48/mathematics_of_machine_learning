# Section 53.9: the selectors of scikit-learn on the training rows.
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, SelectFdr, SelectFpr, SelectFromModel, SelectFwe,
    SelectKBest, SelectPercentile,
    SequentialFeatureSelector as SFS, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from bcdata import load

Xa, Xb, ya, yb, names = load()
Z = StandardScaler().fit_transform(Xa)          # scaled copy
lr = LogisticRegression(max_iter=1000)
l1 = LogisticRegression(l1_ratio=1, C=0.05, solver="liblinear",
                        random_state=0)
rf = RandomForestClassifier(100, random_state=0)
mi = lambda X, y: mutual_info_classif(X, y, random_state=0)
sel = {
    "KBest f_classif k=5": SelectKBest(k=5),
    "KBest mutual info k=5": SelectKBest(mi, k=5),
    "Percentile 20": SelectPercentile(percentile=20),
    "Fpr alpha=1e-3": SelectFpr(alpha=1e-3),
    "Fdr alpha=1e-3": SelectFdr(alpha=1e-3),
    "Fwe alpha=1e-3": SelectFwe(alpha=1e-3),
    "RFE logistic n=5": RFE(lr, n_features_to_select=5),
    "SFS forward n=5": SFS(lr, n_features_to_select=5, cv=5),
    "FromModel L1 C=0.05": SelectFromModel(l1),
    "FromModel forest mean": SelectFromModel(rf),
}
for name, s in sel.items():
    keep = np.flatnonzero(s.fit(Z, ya).get_support())
    print("%-22s %2d  %s" % (name, len(keep),
          keep if len(keep) <= 8 else "(columns 0-29 minus 5)"))

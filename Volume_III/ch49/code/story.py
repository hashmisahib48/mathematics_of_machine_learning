# Which algorithms care about units? Breast cancer, raw vs scaled (49.1).
import json
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
models = {
    "5-NN": KNeighborsClassifier(5),
    "SVC (RBF)": SVC(),
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=5000),
    "MLPClassifier": MLPClassifier((30,), max_iter=2000, random_state=0),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
}
out = {}
print(f"{'model':>20}{'raw':>8}{'scaled':>8}")
for k, m in models.items():
    a = m.fit(Xtr, ytr).score(Xte, yte)
    s = make_pipeline(StandardScaler(), m).fit(Xtr, ytr).score(Xte, yte)
    out[k] = [round(a, 4), round(s, 4)]
    print(f"{k:>20}{a:8.4f}{s:8.4f}")
raw = PCA().fit(Xtr).explained_variance_ratio_[0]
sc = PCA().fit(StandardScaler().fit_transform(Xtr)).explained_variance_ratio_[0]
out["PCA first ratio"] = [round(raw, 4), round(sc, 4)]
print(f"{'PCA, first ratio':>20}{raw:8.4f}{sc:8.4f}")
sd = Xtr.std(axis=0)
print("feature s.d. from", sd.min().round(4), "to", sd.max().round(1))
out["sd_min_max"] = [float(sd.min()), float(sd.max())]
json.dump(out, open("verify/story.json", "w"), indent=1)

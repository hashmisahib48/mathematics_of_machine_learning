# Repeated noise experiments behind Figures 54.0-1 and 54.10-1.
import json, os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"               # small matrices: threads only slow
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

def one(seed, n=60, p=2000, k=10):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p)); y = rng.randint(0, 2, n)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=1000)
    leaky = cross_val_score(clf, SelectKBest(f_classif, k=k).fit_transform(X, y), y, cv=cv).mean()
    honest = cross_val_score(make_pipeline(SelectKBest(f_classif, k=k), clf), X, y, cv=cv).mean()
    return leaky, honest

R = {}
R["hist"] = [one(s) for s in range(100)]
R["vs_p"] = {p: [one(s, p=p) for s in range(24)] for p in (10, 30, 100, 300, 1000, 3000)}
R["vs_n"] = {n: [one(s, n=n, p=1000) for s in range(24)] for n in (30, 60, 120, 240, 480)}
summ = {"hist": {"leaky_mean": float(np.mean([a for a, b in R["hist"]])), "honest_mean": float(np.mean([b for a, b in R["hist"]])),
                 "leaky_min": float(np.min([a for a, b in R["hist"]])), "honest_max": float(np.max([b for a, b in R["hist"]])),
                 "honest_sd": float(np.std([b for a, b in R["hist"]]))}}
for key in ("vs_p", "vs_n"):
    summ[key] = {str(k): [float(np.mean([a for a, b in v])), float(np.mean([b for a, b in v]))] for k, v in R[key].items()}
here = os.path.dirname(os.path.abspath(__file__))
json.dump({"raw": R, "summary": summ}, open(os.path.join(here, "..", "verify", "sweeps.json"), "w"))
print(json.dumps(summ, indent=1))

# Problem 7C.1: where the leak creeps in, and how big it is
import numpy as np
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = StratifiedKFold(5, shuffle=True, random_state=0)
print("== scaler outside and inside the folds (wine, 5 folds)")
X, y = load_wine(return_X_y=True)
clf = LogisticRegression(max_iter=1000)
Xs = StandardScaler().fit_transform(X)            # sees every row
leak = cross_val_score(clf, Xs, y, cv=cv)
pipe = cross_val_score(make_pipeline(StandardScaler(), clf), X, y, cv=cv)
print("leaky :", leak.round(3), " mean", leak.mean().round(4))
print("pipe  :", pipe.round(3), " mean", pipe.mean().round(4))

print("\n== the same mistake with a step that uses the labels")
rng = np.random.default_rng(0)
Xn = rng.normal(size=(60, 2000))                  # pure noise, 2000 features
yn = np.repeat([0, 1], 30)                        # labels unrelated to X
sel = SelectKBest(f_classif, k=20)
Xk = sel.fit_transform(Xn, yn)                    # chosen on all 60 rows
leak = cross_val_score(clf, Xk, yn, cv=cv)
pipe = cross_val_score(make_pipeline(
    SelectKBest(f_classif, k=20), clf), Xn, yn, cv=cv)
print("leaky :", leak.round(3), " mean", leak.mean().round(3))
print("pipe  :", pipe.round(3), " mean", pipe.mean().round(3))

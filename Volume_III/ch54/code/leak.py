import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.RandomState(0)
X = rng.normal(size=(60, 2000))           # 2000 noise features
y = rng.randint(0, 2, 60)                 # labels unrelated to X
cv = StratifiedKFold(5, shuffle=True, random_state=0)
sel = SelectKBest(f_classif, k=10)
clf = LogisticRegression(max_iter=1000)
inner = make_pipeline(sel, clf)
acc = lambda model, X: cross_val_score(model, X, y, cv=cv).mean()

lk = acc(clf, sel.fit_transform(X, y))
sc = acc(inner, StandardScaler().fit_transform(X))
print(f"majority-class share      {np.bincount(y).max() / 60:.3f}")
print(f"selection before the CV   {lk:.3f}")
print(f"scaling before the CV     {sc:.3f}")
print(f"selection inside the CV   {acc(inner, X):.3f}")

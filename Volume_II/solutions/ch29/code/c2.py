# Problem 29C.2: why the level-1 matrix must be out of fold
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(600, 10, n_informative=3, n_redundant=0, flip_y=0.15, random_state=1)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)
def members(): return [("lr", make_pipeline(StandardScaler(), LogisticRegression())), ("rf", RandomForestClassifier(100, random_state=0)), ("nn1", KNeighborsClassifier(1))]
fitted = [(n, m.fit(Xtr, ytr)) for n, m in members()]
print("member   in-sample  5-fold on the same rows  test   gap (in-sample - 5-fold)")
for (n, m), (_, m0) in zip(fitted, members()):
    cvs = cross_val_score(m0, Xtr, ytr, cv=StratifiedKFold(5)).mean()
    print(f"{n:5s}   {m.score(Xtr, ytr):9.4f}  {cvs:22.4f}  {m.score(Xte, yte):.4f}  {m.score(Xtr, ytr) - cvs:8.4f}")
honest = StackingClassifier(members()).fit(Xtr, ytr); leaky = StackingClassifier(fitted, cv="prefit").fit(Xtr, ytr)
print("\nstack        weights (lr, rf, nn1)   train   test")
for name, s in (("out-of-fold", honest), ("in-sample", leaky)):
    print(f"{name:12s} {np.round(s.final_estimator_.coef_.ravel(), 2)}  {s.score(Xtr, ytr):.4f}  {s.score(Xte, yte):.4f}")
from sklearn.model_selection import cross_val_predict
Zl = leaky.transform(Xtr)                       # in-sample answers of the fitted members
Zo = np.column_stack([cross_val_predict(m, Xtr, ytr, cv=StratifiedKFold(5), method="predict_proba")[:, 1] for _, m in members()])
print("\naccuracy of each column of Z used as a vote (lr, rf, nn1), on the training rows:")
print("  in-sample answers :", [round(float(((Zl[:, j] > 0.5) == ytr).mean()), 4) for j in range(3)])
print("  out-of-fold answers:", [round(float(((Zo[:, j] > 0.5) == ytr).mean()), 4) for j in range(3)])
print("correlation of each column with y:")
print("  in-sample :", [round(float(np.corrcoef(Zl[:, j], ytr)[0, 1]), 3) for j in range(3)])
print("  out-of-fold:", [round(float(np.corrcoef(Zo[:, j], ytr)[0, 1]), 3) for j in range(3)])

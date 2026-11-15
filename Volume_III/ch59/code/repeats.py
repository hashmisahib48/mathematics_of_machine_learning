# Section 59.7: repeated cross-validation as a report.
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                             random_state=0)
scaled = lambda m: make_pipeline(StandardScaler(), m)
cands = {"logistic": scaled(LogisticRegression(max_iter=1000)),
         "SVC": scaled(SVC()),
         "forest": RandomForestClassifier(random_state=0)}
S = {k: cross_val_score(m, Xtr, ytr, cv=cv)
     for k, m in cands.items()}
for k, s in S.items():
    rep = s.reshape(10, 5).mean(axis=1)    # one mean per repeat
    print(f"{k:9s}mean {s.mean():.4f}  "
          f"sd of 50 scores {s.std():.4f}  "
          f"sd of 10 repeats {rep.std():.4f}")
d = S["SVC"] - S["logistic"]               # paired by fold
print(f"SVC minus logistic: mean {d.mean():.4f}, SVC better in "
      f"{(d > 0).sum()}, equal in {(d == 0).sum()}, worse in "
      f"{(d < 0).sum()} of 50 folds")

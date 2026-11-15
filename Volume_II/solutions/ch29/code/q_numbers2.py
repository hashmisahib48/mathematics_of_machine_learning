# Numbers quoted in the model answers to Questions 3 and 6 of Chapter 29
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_friedman1
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor,
                              VotingClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.model_selection import (train_test_split, cross_val_score, RepeatedStratifiedKFold, KFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import binomtest

X, y = load_breast_cancer(return_X_y=True)          # 0 malignant, 1 benign
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("svc", make_pipeline(sc(), CalibratedClassifierCV(SVC(), method="sigmoid", cv=5, ensemble=False))),
       ("rf", RandomForestClassifier(100, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]

print("\n== Q3  ten forests that differ only in random_state (friedman1, 1000 rows)")
Xf, yf = make_friedman1(1500, n_features=10, noise=1.0, random_state=0)
Xf1, Xf2, yf1, yf2 = Xf[:750], Xf[750:], yf[:750], yf[750:]
fs = [(f"rf{i}", RandomForestRegressor(100, random_state=i)) for i in range(10)]
one = fs[0][1].fit(Xf1, yf1); print("one forest of 100 trees: test R2", round(one.score(Xf2, yf2), 4))
big = RandomForestRegressor(1000, random_state=0).fit(Xf1, yf1); print("one forest of 1000 trees: test R2", round(big.score(Xf2, yf2), 4))
st = StackingRegressor(fs, cv=5).fit(Xf1, yf1)
print("stack of ten forests: test R2", round(st.score(Xf2, yf2), 4), " weights", st.final_estimator_.coef_.round(3),
      " sum", st.final_estimator_.coef_.sum().round(3), " alpha", st.final_estimator_.alpha_)
Pf = np.array([m.predict(Xf2) for m in st.estimators_])
C = np.corrcoef(Pf - yf2); print("mean correlation of the errors of two forests:", round(C[np.triu_indices(10, 1)].mean(), 4))
print("simple average of the ten: test R2", round(r2_score(yf2, Pf.mean(0)), 4))

print("\n== Q6  ranking committees on 143 test patients")
n = len(yte); a = 0.958
print(f"standard error of an accuracy of {a}: {np.sqrt(a * (1 - a) / n):.4f};  one patient = {1 / n:.4f} = {1 / n / np.sqrt(a * (1 - a) / n):.2f} standard errors")
hv = VotingClassifier(est).fit(Xtr, ytr); sv = VotingClassifier(est, voting="soft").fit(Xtr, ytr); sk = StackingClassifier(est).fit(Xtr, ytr)
C3 = {"hard": hv, "soft": sv, "stack": sk}
W = {k: m.predict(Xte) != yte for k, m in C3.items()}
for k, v in W.items(): print(f"{k:6s} errors {int(v.sum())}")
for a_, b_ in (("hard", "soft"), ("hard", "stack"), ("soft", "stack")):
    b01 = int((W[a_] & ~W[b_]).sum()); b10 = int((~W[a_] & W[b_]).sum())
    print(f"{a_} vs {b_}: patients where only {a_} errs {b01}, only {b_} errs {b10}; exact McNemar p = "
          f"{binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0:.3f}")
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
sc_ = {k: cross_val_score(m, Xtr, ytr, cv=cv) for k, m in C3.items()}
for k, v in sc_.items(): print(f"{k:6s} 10 folds: mean {v.mean():.4f}")
for a_, b_ in (("soft", "hard"), ("stack", "hard")):
    d = sc_[a_] - sc_[b_]; print(f"paired difference {a_} - {b_}: mean {d.mean():.4f}, standard error {d.std(ddof=1) / np.sqrt(len(d)):.4f}")

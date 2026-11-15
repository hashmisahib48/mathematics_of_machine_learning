# Numbers quoted in the model answers to Questions 1, 2, 4, 5 of Chapter 29 (Questions 3 and 6: q_numbers2.py)
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

print("== Q1  ties and the cost of a missed tumour (soft vote of the four members)")
vote = VotingClassifier(est, voting="soft").fit(Xtr, ytr)
pm = vote.predict_proba(Xte)[:, 0]                   # probability of malignant (class 0)
for t in (0.5, 0.2, 0.09):
    pred_mal = pm >= t
    fn = int(((~pred_mal) & (yte == 0)).sum()); fp = int((pred_mal & (yte == 1)).sum())
    print(f"call malignant when P(malignant) >= {t:4.2f}: missed malignant {fn}, false alarms {fp}")
print("threshold for cost ratio 10:1 (missed : false alarm):", round(1 / (1 + 10), 3))
hard = VotingClassifier(est).fit(Xtr, ytr)
votes_mal = np.array([m.predict(Xte) == 0 for m in hard.estimators_]).sum(0)
for k in range(5): print(f"patients with {k} malignant votes: {int((votes_mal == k).sum())}  of which truly malignant {int(((votes_mal == k) & (yte == 0)).sum())}")
print("if the labels were swapped (benign = 0) the 2-2 ties would go to benign: same 3 patients,",
      "truly malignant among them:", int(((votes_mal == 2) & (yte == 0)).sum()))

print("\n== Q2  when does a weak member help? (diabetes split of regress.py)")
Xd, yd = load_diabetes(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.25, random_state=0)
names = ["ols", "forest", "boost"]
ms = [LinearRegression(), RandomForestRegressor(300, max_features=0.33, min_samples_leaf=5, random_state=0),
      GradientBoostingRegressor(random_state=0)]
P = np.array([m.fit(Xa, ya).predict(Xb) for m in ms])
for n, p in zip(names, P): print(f"{n:7s} R2 {r2_score(yb, p):.4f}  mean error {np.mean(p - yb):7.2f}")
for i in range(3):
    for j in range(i + 1, 3):
        print(f"average of {names[i]:6s} and {names[j]:6s}: R2 {r2_score(yb, (P[i] + P[j]) / 2):.4f}"
              f"   correlation of their errors {np.corrcoef(P[i] - yb, P[j] - yb)[0, 1]:.3f}")
print("average of all three: R2", round(r2_score(yb, P.mean(0)), 4))
e1, e2 = 1.0, -2.5
print(f"two members with errors {e1} and {e2}: average error {(e1 + e2) / 2}, squared {((e1 + e2) / 2)**2} < {e1**2}")
print("an average of two members beats the first when e2 lies between", -3 * e1, "and", e1, "(e1 = 1)")

print("\n== Q4  members trained on (K-1)/K of the rows against all rows")
def acc_by_fraction(n, fracs, reps=4):
    out = {f: [] for f in fracs}
    for s in range(reps):
        Xs, ys = make_classification(n + 4000, 20, n_informative=6, flip_y=0.1, random_state=s)
        Xt, yt, Xv, yv = Xs[:n], ys[:n], Xs[n:], ys[n:]
        r = np.random.default_rng(s)
        for f in fracs:
            idx = r.permutation(n)[:int(round(f * n))]
            out[f].append(RandomForestClassifier(50, random_state=0).fit(Xt[idx], yt[idx]).score(Xv, yv))
    return {f: np.mean(v) for f, v in out.items()}
for n in (100, 2000):
    a = acc_by_fraction(n, (0.5, 0.8, 0.98, 1.0))
    print(f"n = {n:4d}: accuracy with all rows {a[1.0]:.4f};  with 1/2 (K=2) {a[0.5]:.4f}, 4/5 (K=5) {a[0.8]:.4f}, 49/50 (K=50) {a[0.98]:.4f};"
          f"  loss {a[1.0] - a[0.5]:.4f}, {a[1.0] - a[0.8]:.4f}, {a[1.0] - a[0.98]:.4f}")

print("\n== Q5  the final estimator")
from sklearn.compose import ColumnTransformer
Xx = np.random.default_rng(0).uniform(-1, 1, (2000, 2)); yx = ((Xx[:, 0] > 0) & (Xx[:, 1] > 0)).astype(int)   # AND
def one_feat(j): return make_pipeline(ColumnTransformer([("c", "passthrough", [j])]), LogisticRegression(C=100))
two = [("a", one_feat(0)), ("b", one_feat(1))]
Xx1, Xx2, yx1, yx2 = Xx[:1000], Xx[1000:], yx[:1000], yx[1000:]
print("logistic regression on both features:", round(LogisticRegression().fit(Xx1, yx1).score(Xx2, yx2), 4))
for nm, fe in (("logistic", LogisticRegression()), ("forest", RandomForestClassifier(100, random_state=0))):
    s = StackingClassifier(two, final_estimator=fe, cv=5).fit(Xx1, yx1)
    print(f"AND data, two one-feature linear members, {nm:8s} final estimator: test accuracy {s.score(Xx2, yx2):.4f}")
Xs, ys = make_classification(600, 10, n_informative=3, n_redundant=0, flip_y=0.15, random_state=1)
Xs1, Xs2, ys1, ys2 = train_test_split(Xs, ys, test_size=0.5, random_state=0, stratify=ys)
mem = [("lr", make_pipeline(sc(), LogisticRegression())), ("rf", RandomForestClassifier(100, random_state=0)), ("knn", KNeighborsClassifier(15))]
for nm, fe in (("logistic", LogisticRegression()), ("tree", DecisionTreeClassifier(random_state=0)), ("forest", RandomForestClassifier(100, random_state=0))):
    s = StackingClassifier(mem, final_estimator=fe, cv=5)
    cv = cross_val_score(s, Xs1, ys1, cv=5).mean(); s.fit(Xs1, ys1)
    print(f"noisy data, flexible members, {nm:8s} final estimator: 5-fold {cv:.4f}  test {s.score(Xs2, ys2):.4f}")


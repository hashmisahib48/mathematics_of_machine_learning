# Problem 25B.3: class weights in a random forest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=4,
    weights=[0.93, 0.07], flip_y=0.02, class_sep=0.8, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)
print("minority samples: %d of %d for training, %d of %d for test" % (
    y_tr.sum(), len(y_tr), y_te.sum(), len(y_te)))
minority = np.where(y_tr == 1)[0]
print("class_weight         recall precision  mean P  in bag")
for cw in (None, "balanced", "balanced_subsample"):
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                class_weight=cw, random_state=0)
    rf.fit(X_tr, y_tr)
    p1 = rf.predict_proba(X_te)[:, 1]
    pred = rf.predict(X_te)
    inbag = np.mean([np.isin(minority, s).mean()
                     for s in rf.estimators_samples_])
    print("%-18s %7.3f %9.3f %7.3f %7.3f" % (
        cw, recall_score(y_te, pred), precision_score(y_te, pred),
        p1.mean(), inbag))
    if cw is None:
        base = p1
print("\nclass_weight=None with a lower threshold on P(minority):")
for t in (0.3, 0.2, 0.15, 0.1):
    pred = (base > t).astype(int)
    print("  t = %.2f: recall %.3f, precision %.3f, alarms %d" % (
        t, recall_score(y_te, pred), precision_score(y_te, pred),
        pred.sum()))

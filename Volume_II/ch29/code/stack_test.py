# The test: MyStack and two voting rules against scikit-learn.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from stack_scratch import MyStack

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("svc", make_pipeline(sc(), SVC())),
       ("rf", RandomForestClassifier(50, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]

mine = MyStack([m for _, m in est]).fit(Xtr, ytr)
ref = StackingClassifier(est).fit(Xtr, ytr)
print("methods:", mine.methods_ == ref.stack_method_, mine.methods_)
for name, a, b in (
        ("transform", mine.transform(Xte), ref.transform(Xte)),
        ("predict_proba", mine.predict_proba(Xte),
         ref.predict_proba(Xte))):
    print(f"{name:14s} max |difference| {np.abs(a - b).max():.1e}")
    assert np.allclose(a, b)
assert (mine.predict(Xte) == ref.predict(Xte)).all()
print("stacking test accuracy", round(ref.score(Xte, yte), 4))

trio = [e for e in est if e[0] != "svc"]   # predict_proba
hard = VotingClassifier(trio).fit(Xtr, ytr)
soft = VotingClassifier(trio, voting="soft").fit(Xtr, ytr)
votes = np.array([m.predict(Xte) for m in hard.estimators_])
probs = np.mean([m.predict_proba(Xte) for m in soft.estimators_], 0)
assert (hard.predict(Xte) == (votes.sum(0) > 1.5)).all()   # 29.3-1
assert (soft.predict(Xte) == probs.argmax(1)).all()        # 29.3-2
diff = (hard.predict(Xte) != soft.predict(Xte)).sum()
print("voting agrees with NumPy; hard and soft differ on", diff,
      "of", len(yte), "patients")

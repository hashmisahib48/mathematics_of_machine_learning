# Who is wrong? Overlap of the test errors of the four members.
import json
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("svc", make_pipeline(sc(), CalibratedClassifierCV(
           SVC(), method="sigmoid", cv=5, ensemble=False))),
       ("rf", RandomForestClassifier(100, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]
vote = VotingClassifier(est).fit(Xtr, ytr)
wrong = np.array([m.predict(Xte) != yte for m in vote.estimators_])
names = [n for n, _ in est]
print("patients misclassified by at least one member:",
      int(wrong.any(0).sum()), " by all four:",
      int(wrong.all(0).sum()))
print("shared errors (diagonal: errors of the member)")
print(np.array([[int((a & b).sum()) for b in wrong]
                for a in wrong]))
print("correlation of the error indicators, lr/svc/rf/knn:")
print(np.corrcoef(wrong).round(2))
n_right = (~wrong).sum(0)
print("members right per patient, 0..4:",
      np.bincount(n_right, minlength=5))
tie = n_right == 2
print("2-2 ties among the", len(yte), "patients:", int(tie.sum()),
      " truly malignant:", int((yte[tie] == 0).sum()),
      " (a tie goes to class", vote.classes_[0], "= malignant)")
for nm, m in (("soft vote", VotingClassifier(est, voting="soft")),
              ("stack", StackingClassifier(est))):
    ok = m.fit(Xtr, ytr).predict(Xte)[tie] == yte[tie]
    print(f"tied patients decided correctly by {nm}:", int(ok.sum()),
          "of", int(tie.sum()))
json.dump({"wrong": wrong.astype(int).tolist(), "names": names,
           "y": yte.tolist()}, open("verify/case_errors.json", "w"))

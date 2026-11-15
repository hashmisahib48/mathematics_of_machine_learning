# Problem 29B.2: a fifth voter for the committee of Listing 29.2
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as sc
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)   # 0 malignant, 1 benign
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("svc", make_pipeline(sc(), CalibratedClassifierCV(
           SVC(), method="sigmoid", cv=5, ensemble=False))),
       ("rf", RandomForestClassifier(100, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]
v4 = VotingClassifier(est).fit(Xtr, ytr)
five = est + [("gb", GBC(random_state=0))]
v5 = VotingClassifier(five).fit(Xtr, ytr)
p4, p5 = v4.predict(Xte), v5.predict(Xte)
gb = v5.estimators_[4].predict(Xte)
print("test errors: four voters", int((p4 != yte).sum()),
      " five voters", int((p5 != yte).sum()),
      " boosting alone", int((gb != yte).sum()))
d = np.where(p4 != p5)[0]
print("patients on which the two votes differ:", d.tolist())
ok4 = np.array([m.predict(Xte) == yte for m in v4.estimators_])
tie = ok4.sum(0) == 2                       # 2-2 among the four
print("2-2 ties of the four voters:", np.where(tie)[0].tolist())
for i in np.where(tie)[0]:
    print(f"  patient {i}: true class {yte[i]}, four-vote {p4[i]},"
          f" boosting {gb[i]}, five-vote {p5[i]}")

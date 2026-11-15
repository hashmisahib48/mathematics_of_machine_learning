# class_weight multiplies C of the samples of a class (Eq. 22.4-9).
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_classification(
    n_samples=2000, n_features=6, n_informative=3, n_redundant=0,
    weights=[0.9, 0.1], class_sep=1.0, flip_y=0.02, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)
sc = StandardScaler().fit(X_tr)
Z_tr, Z_te = sc.transform(X_tr), sc.transform(X_te)

for cw in (None, "balanced"):
    m = SVC(C=1.0, class_weight=cw).fit(Z_tr, y_tr)
    pr = m.predict(Z_te)
    print(f"class_weight={cw}: class_weight_ =",
          np.round(m.class_weight_, 3))
    print(f"   recall {recall_score(y_te, pr):.3f}   precision "
          f"{precision_score(y_te, pr):.3f}")

a = SVC(class_weight={0: 1.0, 1: 5.0}).fit(Z_tr, y_tr)
sw = np.where(y_tr == 1, 5.0, 1.0)
b = SVC().fit(Z_tr, y_tr, sample_weight=sw)
print("class_weight {0: 1, 1: 5} = sample_weight 5 on class 1:",
      np.allclose(a.dual_coef_, b.dual_coef_))
print("largest |alpha|:", np.abs(a.dual_coef_).max().round(3),
      "(the bound is 5 C = 5)")

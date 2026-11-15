# RidgeClassifierCV: the penalty chosen by leave-one-out (§17.10).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

alphas = np.logspace(-2, 3, 6)          # 0.01 ... 1000
cv = RidgeClassifierCV(alphas=alphas, store_cv_results=True)
cv.fit(Xtr, ytr)
loo = cv.cv_results_.mean(axis=(0, 1))   # mean LOO squared error
print("alpha     ", alphas.tolist())
print("LOO error ", loo.round(4).tolist())
print("alpha_ =", cv.alpha_, " best_score_ =",
      round(cv.best_score_, 4))

ref = RidgeClassifier(alpha=cv.alpha_).fit(Xtr, ytr)
same = np.allclose(cv.coef_, ref.coef_)
print("same weights as RidgeClassifier(alpha_):", same)
print("test accuracy:", round(cv.score(Xte, yte), 4))

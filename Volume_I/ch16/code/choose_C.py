# Choosing C by cross-validation, two ways (breast-cancer data).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
Cs = np.logspace(-4, 2, 13)                    # 1e-4 ... 1e2

pipe = make_pipeline(StandardScaler(), LogisticRegression())
tr, va = validation_curve(
    pipe, X_tr, y_tr, cv=5, scoring="neg_log_loss",
    param_name="logisticregression__C", param_range=Cs)
print("        C   train   validation   (mean log loss, 5 folds)")
for C, a, b in zip(Cs, -tr.mean(1), -va.mean(1)):
    print(f"{C:9.4f} {a:7.4f} {b:10.4f}")

lrcv = LogisticRegressionCV(
    Cs=Cs, cv=5, scoring="neg_log_loss", l1_ratios=(0.0,),
    use_legacy_attributes=False)
cv = make_pipeline(StandardScaler(), lrcv).fit(X_tr, y_tr)
curve = -lrcv.scores_[:, 0, :].mean(0)         # (folds, l1, Cs)
best = Cs[np.argmax(va.mean(1))]
print("best C, validation_curve :", round(best, 4))
print("C_ of LogisticRegressionCV:", round(lrcv.C_, 4))
print("largest gap between the curves:",
      round(float(abs(curve + va.mean(1)).max()), 4))
print("scores_ has shape", lrcv.scores_.shape)
acc = (cv.predict(X_te) == y_te).mean()
print("test accuracy:", round(float(acc), 4))

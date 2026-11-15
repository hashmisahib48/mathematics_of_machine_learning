# Case study: tumours, LDA and QDA against logistic regression.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis as LDA,
    QuadraticDiscriminantAnalysis as QDA)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import (
    cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    d.data, d.target, test_size=0.25, random_state=0,
    stratify=d.target)
Z = StandardScaler().fit_transform(X_tr)
ev = np.linalg.eigvalsh(np.cov(Z[y_tr == 1].T, bias=True))
print(f"class 1: smallest eigenvalue {ev[0]:.1e}")
try:
    make_pipeline(StandardScaler(), QDA()).fit(X_tr, y_tr)
except np.linalg.LinAlgError as err:
    print("QDA():", str(err)[:50], "...")

models = {"LDA": LDA(),
          "LDA lsqr, auto": LDA(solver="lsqr", shrinkage="auto"),
          "QDA, reg_param 0.1": QDA(reg_param=0.1),
          "logistic": LogisticRegression()}
print("model               train   test  5-fold  log loss")
for name, m in models.items():
    m = models[name] = make_pipeline(StandardScaler(), m)
    m.fit(X_tr, y_tr)
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    ll = log_loss(y_te, m.predict_proba(X_te))
    print(f"{name:19s} {m.score(X_tr, y_tr):.4f} "
          f"{m.score(X_te, y_te):.4f} {cv:.4f}  {ll:.4f}")
pred = models["LDA"].predict(X_te)
print(confusion_matrix(y_te, pred))      # rows true: malignant, benign

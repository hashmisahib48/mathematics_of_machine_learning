# Case study: tumours from 30 measurements.
import warnings
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target        # 0 malignant, 1 benign
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("train", X_tr.shape, " test", X_te.shape,
      " benign share", round(y.mean(), 3))

base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print("baseline test accuracy:", round(base.score(X_te, y_te), 4))

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    raw = LogisticRegression().fit(X_tr, y_tr)
print("raw features: n_iter_", raw.n_iter_[0], " warnings",
      len(rec), " test accuracy", round(raw.score(X_te, y_te), 4))

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_tr, y_tr)
p_te = pipe.predict_proba(X_te)
pred = pipe.predict(X_te)
print("scaled: n_iter_", pipe[-1].n_iter_[0],
      " train", round(pipe.score(X_tr, y_tr), 4),
      " test", round(pipe.score(X_te, y_te), 4),
      " test log loss", round(log_loss(y_te, p_te), 4))
print("confusion matrix (rows true, columns predicted;")
print("order: malignant, benign)")
print(confusion_matrix(y_te, pred))
print("precision, recall of malignant:",
      round(precision_score(y_te, pred, pos_label=0), 4),
      round(recall_score(y_te, pred, pos_label=0), 4))
cv = cross_val_score(pipe, X_tr, y_tr, cv=5)
print("5-fold accuracy:", np.round(cv, 3),
      " mean", round(cv.mean(), 4))

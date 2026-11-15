# 8.7  The same protocol for a classifier (breast cancer data).
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("train", len(y_tr), " test", len(y_te))
model = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000))
for name, m in (("majority class", DummyClassifier()),
                ("scaled logistic", model)):
    r = cross_validate(m, X_tr, y_tr, cv=5,
                       scoring=["accuracy", "f1"])
    print(f"{name:16s} CV accuracy {r['test_accuracy'].mean():.3f}"
          f"   CV F1 {r['test_f1'].mean():.3f}")
model.fit(X_tr, y_tr)                  # refit on training part
print("test accuracy", round(model.score(X_te, y_te), 3),
      "  F1", round(f1_score(y_te, model.predict(X_te)), 3))

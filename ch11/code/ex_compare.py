# The classifiers of this book compared on the breast cancer data.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)
models = {"k-NN, k = 7": make_pipeline(StandardScaler(),
                                       KNeighborsClassifier(7)),
          "logistic": make_pipeline(StandardScaler(),
                                    LogisticRegression()),
          "naive Bayes": GaussianNB(),
          "tree, depth 4": DecisionTreeClassifier(max_depth=4,
                                                  random_state=0)}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
for name, m in models.items():
    r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
    acc, auc = r["test_accuracy"], r["test_roc_auc"]
    print(f"{name:14s} accuracy {acc.mean():.3f} ± {acc.std():.3f}"
          f"   AUC {auc.mean():.3f}")

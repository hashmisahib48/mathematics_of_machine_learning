import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     cross_val_score)
from sklearn.naive_bayes import (GaussianNB, MultinomialNB,
    ComplementNB, BernoulliNB, CategoricalNB)

X, y = load_digits(return_X_y=True)      # 1797 images, 64 pixels
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
grid = {"var_smoothing": np.logspace(-9, 0, 10)}
gs = GridSearchCV(GaussianNB(), grid, cv=5).fit(Xtr, ytr)
print("best var_smoothing:", gs.best_params_["var_smoothing"])

models = {"baseline (most frequent)": DummyClassifier(),
          "GaussianNB, default": GaussianNB(),
          "GaussianNB, tuned": gs.best_estimator_,
          "MultinomialNB": MultinomialNB(),
          "ComplementNB": ComplementNB(),
          "BernoulliNB (binarize=8)": BernoulliNB(binarize=8),
          "CategoricalNB (17 levels)":
              CategoricalNB(min_categories=17)}
print(f"{'model':26s} train   test  5-fold")
for name, m in models.items():
    m.fit(Xtr, ytr)
    cv = cross_val_score(m, Xtr, ytr, cv=5).mean()
    tr, te = m.score(Xtr, ytr), m.score(Xte, yte)
    print(f"{name:26s} {tr:.3f}  {te:.3f}  {cv:.3f}")

for name in ["GaussianNB, default", "MultinomialNB"]:
    m = models[name]
    conf = m.predict_proba(Xte).max(axis=1).mean()
    acc = m.score(Xte, yte)
    print(f"{name:20s} mean confidence {conf:.3f}, "
          f"accuracy {acc:.3f}")

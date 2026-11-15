# Section 59.4: breast cancer, from a baseline to a chosen model.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
print("training part", Xtr.shape, "class counts", np.bincount(ytr))

scaled = lambda m: make_pipeline(StandardScaler(), m)
models = {
    "majority class": DummyClassifier(strategy="most_frequent"),
    "logistic": scaled(LogisticRegression(max_iter=1000)),
    "SVC": scaled(SVC()),
    "random forest": RandomForestClassifier(random_state=0),
    "hist. boosting": HistGradientBoostingClassifier(
        random_state=0),
}
for name, m in models.items():       # the test rows are not touched
    s = cross_val_score(m, Xtr, ytr, cv=5)
    print(f"{name:15s}{s.mean():.4f} +- {s.std():.4f}"
          f"  ({s.min():.3f} to {s.max():.3f})")

grid = {"svc__C": 2.0 ** np.arange(-5, 10, 2),
        "svc__gamma": 2.0 ** np.arange(-15, 2, 2)}
search = GridSearchCV(scaled(SVC()), grid, cv=5).fit(Xtr, ytr)
bp = search.best_params_
g2 = np.log2(bp["svc__gamma"])
print(f"grid cells {len(search.cv_results_['params'])}, "
      f"best C = {bp['svc__C']:.0f}, gamma = 2^{g2:.0f}")
print(f"best 5-fold accuracy {search.best_score_:.4f}")

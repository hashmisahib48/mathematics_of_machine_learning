# Breast cancer: boosting against logistic regression (Chapter 16).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0,
    stratify=data.target)


def report(name, model):
    cv = cross_val_score(model, Xtr, ytr, cv=5).mean()
    model.fit(Xtr, ytr)
    ll = log_loss(yte, model.predict_proba(Xte))
    acc = model.score(Xte, yte)
    print(f"{name:21s} cv {cv:.4f} test {acc:.4f} log loss {ll:.4f}")
    return model


report("logistic regression",
       make_pipeline(StandardScaler(), LogisticRegression()))
report("one full tree", DecisionTreeClassifier(random_state=0))
default = report("boosting, defaults", GBC(random_state=0))
report("depth 1 (stumps)", GBC(max_depth=1, random_state=0))
report("exponential loss", GBC(loss="exponential", random_state=0))
report("subsample 0.5", GBC(subsample=0.5, random_state=0))
stop = GBC(n_estimators=1000, n_iter_no_change=10, random_state=0)
report("early stopping", stop)
print("  trees kept by early stopping:", stop.n_estimators_)
imp = default.feature_importances_
top = np.argsort(imp)[::-1][:3]
print("largest importances:")
for k in top:
    print(f"  {data.feature_names[k]:22s} {imp[k]:.3f}")

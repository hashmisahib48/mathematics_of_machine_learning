# Case study: forests against one tree and logistic regression.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)   # 0 malignant, 1 benign
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
rs = dict(random_state=0)
models = {
    "one tree": DecisionTreeClassifier(**rs),
    "bagged trees": BaggingClassifier(
        DecisionTreeClassifier(**rs), n_estimators=100,
        oob_score=True, **rs),
    "random forest": RandomForestClassifier(
        n_estimators=100, oob_score=True, **rs),
    "extra trees": ExtraTreesClassifier(n_estimators=100, **rs),
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression()),
}
print("model                 5-fold    test     OOB")
for name, m in models.items():
    cv = cross_val_score(m, X_tr, y_tr, cv=5).mean()
    m.fit(X_tr, y_tr)
    oob = getattr(m, "oob_score_", None)
    oob = "    -" if oob is None else f"{oob:.4f}"
    print(f"{name:20s}  {cv:.4f}  {m.score(X_te, y_te):.4f}  {oob}")

bag_lr = BaggingClassifier(
    make_pipeline(StandardScaler(), LogisticRegression()),
    n_estimators=100, **rs).fit(X_tr, y_tr)
print("bagged logistic regression test:",
      round(bag_lr.score(X_te, y_te), 4))
rf = models["random forest"]
top = np.argsort(rf.feature_importances_)[::-1][:5]
names = load_breast_cancer().feature_names
for j in top:
    print(f"  {names[j]:22s} {rf.feature_importances_[j]:.4f}")

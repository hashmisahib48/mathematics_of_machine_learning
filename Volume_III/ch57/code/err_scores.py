# Section 57.8: three common errors with scores.
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(Xtr, ytr)
print("AUC of the scores :", round(roc_auc_score(
    yte, lr.decision_function(Xte)), 4))
print("AUC of the labels :", round(roc_auc_score(
    yte, lr.predict(Xte)), 4), "(no error, no warning)")

Xi, yi = load_iris(return_X_y=True)
svc = SVC().fit(Xi, yi)
P = LogisticRegression(max_iter=500).fit(Xi, yi).predict_proba(Xi)
calls = {
    "f1, three classes": lambda: f1_score(yi, svc.predict(Xi)),
    "AUC, no multi_class": lambda: roc_auc_score(yi, P),
    "AUC, margins": lambda: roc_auc_score(
        yi, svc.decision_function(Xi), multi_class="ovr"),
    "AUC, ovr": lambda: roc_auc_score(yi, P, multi_class="ovr")}
for name, call in calls.items():
    try:
        print(f"{name:20s}", round(call(), 4))
    except ValueError as e:
        print(f"{name:20s}ValueError:", str(e)[:48])

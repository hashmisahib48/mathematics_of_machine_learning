# Section 57.8: scorers, make_scorer and the sign convention.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (fbeta_score, get_scorer_names,
                             log_loss, make_scorer)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)       # 0 = malignant
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)


def cost(y_true, y_pred):        # 10 per miss, 1 per alarm
    miss = ((y_true == 0) & (y_pred == 1)).sum()
    alarm = ((y_true == 1) & (y_pred == 0)).sum()
    return (10 * miss + alarm) / len(y_true)


scoring = {
    "auc": "roc_auc",                 # class 1 is positive
    "f2_malignant": make_scorer(fbeta_score, beta=2, pos_label=0),
    "nll": make_scorer(log_loss, greater_is_better=False,
                       response_method="predict_proba"),
    "cost": make_scorer(cost, greater_is_better=False),
}
pipe = make_pipeline(StandardScaler(), LogisticRegression())
cv = cross_validate(pipe, Xtr, ytr, cv=5, scoring=scoring)
print(len(get_scorer_names()), "scorer names; cross_validate keys:")
print(" ", sorted(k for k in cv if k.startswith("test_")))
for k in scoring:
    print(f"{k:13s} mean {cv['test_' + k].mean():8.4f}")
pipe.fit(Xtr, ytr)
print("cost scorer:", round(scoring["cost"](pipe, Xte, yte), 4),
      "= -cost:", round(-cost(yte, pipe.predict(Xte)), 4))

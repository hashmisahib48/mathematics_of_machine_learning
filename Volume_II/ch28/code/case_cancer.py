# Case study: HistGradientBoostingClassifier on breast cancer.
import os
os.environ["OMP_NUM_THREADS"] = "1"       # small data: one thread
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV, train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

default = HGB(random_state=0).fit(X_tr, y_tr)
grid = {"max_leaf_nodes": [4, 8, 31],
        "min_samples_leaf": [5, 10, 20],
        "l2_regularization": [0, 1, 10]}
search = GridSearchCV(HGB(random_state=0), grid, cv=5,
                      scoring="neg_log_loss").fit(X_tr, y_tr)

tuned = search.best_estimator_
for name, m in (("default", default), ("tuned", tuned)):
    p = m.predict_proba(X_te)
    print("%-8s train %.4f  test %.4f  log loss %.4f  trees %d" % (
        name, m.score(X_tr, y_tr), m.score(X_te, y_te),
        log_loss(y_te, p), m.n_iter_))
print("best:", search.best_params_)
print("5-fold log loss of the best: %.4f" % -search.best_score_)

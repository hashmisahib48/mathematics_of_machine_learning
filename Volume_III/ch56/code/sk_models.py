# Section 56.7: one grid over three kinds of model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import cancer_split

Xtr, Xte, ytr, yte = cancer_split()
pipe = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
models = [{"clf": [SVC()], "clf__C": [1, 10, 100]},
          {"clf": [LogisticRegression(max_iter=1000)],
           "clf__C": [0.01, 0.1, 1]},
          {"clf": [RandomForestClassifier(random_state=0)],
           "clf__max_depth": [3, None]}]
gm = GridSearchCV(pipe, models, cv=5).fit(Xtr, ytr)
best = {}
for p, s in zip(gm.cv_results_["params"],
                gm.cv_results_["mean_test_score"]):
    k = type(p["clf"]).__name__
    best[k] = max(best.get(k, 0), s)
for k, s in best.items():
    print(f"{k:24s}{s:.4f}")
print("chosen:", type(gm.best_params_["clf"]).__name__,
      gm.best_params_["clf__C"], " test", round(gm.score(Xte, yte), 4))

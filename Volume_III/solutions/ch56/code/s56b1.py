# Problem 56B.1: the one-standard-error rule as a refit callable
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.25,
                                  random_state=0, stratify=y)
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000))
Cs = np.logspace(-4, 2, 13)
grid = {"logisticregression__C": Cs}

def one_se(res):            # res = cv_results_ -> an index
    m, s = res["mean_test_score"], res["std_test_score"]
    b = int(np.argmax(m))
    ok = np.flatnonzero(m >= m[b] - s[b] / np.sqrt(5))
    return int(ok[np.argmin(Cs[ok])])     # smallest C that is good

g1 = GridSearchCV(pipe, grid, cv=5).fit(Xa, ya)
g2 = GridSearchCV(pipe, grid, cv=5, refit=one_se).fit(Xa, ya)
r, b = g1.cv_results_, g1.best_index_
se = r["std_test_score"][b] / np.sqrt(5)
print("best: mean %.4f  std/sqrt(5) %.4f  threshold %.4f"
      % (r["mean_test_score"][b], se, r["mean_test_score"][b] - se))
print("      C       mean    std")
for i, C in enumerate(Cs):
    tag = "best" if i == b else "rule" if i == g2.best_index_ else ""
    print("%9.4f  %.4f  %.4f  %s" % (C, r["mean_test_score"][i],
                                      r["std_test_score"][i], tag))
for name, g in (("best", g1), ("rule", g2)):
    i = g.best_index_
    print("%s: C = %.4f  CV %.4f  test %.4f" % (name, Cs[i],
          g.cv_results_["mean_test_score"][i], g.score(Xt, yt)))
print("rule has best_score_:", hasattr(g2, "best_score_"))

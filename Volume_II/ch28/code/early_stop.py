# Early stopping in HistGradientBoostingClassifier (Section 28.8).
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier as HGB

for n in (2000, 12000):
    X, y = make_classification(n, 20, n_informative=6, flip_y=0.1,
                               random_state=0)
    m = HGB(max_iter=500, random_state=0).fit(X, y)
    print("n = %5d  early stopping on: %-5s  trees: %3d  scores: %d"
          % (n, m.do_early_stopping_, m.n_iter_,
             len(m.validation_score_)))

v = -m.validation_score_                    # validation log loss
best = int(np.argmin(v))
print("log loss: start %.4f, best %.4f after tree %d, last %.4f"
      % (v[0], v[best], best, v[-1]))
print("stopped %d trees after the best one (n_iter_no_change = 10)"
      % (len(v) - 1 - best))

m = HGB(max_iter=500, early_stopping=True, scoring="accuracy",
        validation_fraction=0.2, random_state=0).fit(X, y)
print("scoring='accuracy': trees %d, validation accuracy %.4f"
      % (m.n_iter_, m.validation_score_[-1]))

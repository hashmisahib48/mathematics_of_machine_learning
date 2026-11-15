# Problem 31B.3: what early stopping reports.
import warnings
import numpy as np
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
warnings.simplefilter("ignore", ConvergenceWarning)

X, y = make_moons(100, noise=0.3, random_state=0)
m = MLPClassifier((50,), early_stopping=True, n_iter_no_change=5,
                  validation_fraction=0.2, alpha=0, random_state=0,
                  max_iter=500).fit(X, y)
vs = np.array(m.validation_scores_)
best = int(vs.argmax()) + 1               # epochs are counted from 1
print("n_iter_               :", m.n_iter_)
print("epoch of the best score:", best)
print("best_validation_score_ :", m.best_validation_score_)
print("n_iter_ - best epoch   :", m.n_iter_ - best,
      "= n_iter_no_change + 1")
print("validation rows        :", int(round(0.2 * len(X))),
      "  best score x rows =", m.best_validation_score_ * 20)
print("training rows          :", len(X) - 20,
      "  t_ =", m.t_, "= n_iter_ x rows:", m.n_iter_ * (len(X) - 20))
print("best_loss_             :", m.best_loss_)
print("validation scores of epochs 1 to 18:")
print(vs.round(2).tolist())

# Five classifiers on one data set: how reliable are they?
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

rng = np.random.default_rng(0)            # 3 signals, seen 4 times
Z = rng.normal(size=(25000, 3))
X = np.hstack([Z] + [Z + 0.3 * rng.normal(size=Z.shape)
                     for _ in range(3)])
p_true = 1 / (1 + np.exp(-Z @ [1.6, 1.28, 0.96]))
y = (rng.random(25000) < p_true).astype(int)
X_tr, y_tr, X_te, y_te = X[:5000], y[:5000], X[15000:], y[15000:]

models = {"GaussianNB": GaussianNB(),
          "AdaBoost": AdaBoostClassifier(random_state=0),
          "LinearSVC": LinearSVC(),
          "LogisticRegression": LogisticRegression(),
          "RandomForest": RandomForestClassifier(random_state=0)}
print(f"{'model':19s} log loss  Brier  min p  max p")
for name, m in models.items():
    m.fit(X_tr, y_tr)
    if name == "LinearSVC":               # scores, squashed
        p = 1 / (1 + np.exp(-m.decision_function(X_te)))
    else:
        p = m.predict_proba(X_te)[:, 1]
    print(f"{name:19s} {log_loss(y_te, p):8.4f} "
          f"{brier_score_loss(y_te, p):6.4f} {p.min():6.3f} "
          f"{p.max():6.3f}")
    if name == "GaussianNB":
        t, q = calibration_curve(y_te, p, n_bins=5)
        print("   five bins, predicted:", q.round(2))
        print("   five bins, observed :", t.round(2))

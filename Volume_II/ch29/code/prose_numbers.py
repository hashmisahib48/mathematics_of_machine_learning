# Numbers quoted in the text that no other program prints.
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from condorcet import majority

x = np.arange(1.0, 7.0); y = np.array([2.0, 5.0, 3.0, 8.0, 7.0, 11.0])
print("Example 29.5-2")
oof = np.zeros((6, 2))
for held in ([0, 1], [2, 3], [4, 5]):
    keep = np.setdiff1d(np.arange(6), held)
    xs, ys = x[keep], y[keep]
    w = np.sum((xs - xs.mean()) * (ys - ys.mean())) / np.sum(
        (xs - xs.mean()) ** 2)
    b = ys.mean() - w * xs.mean()
    print(f"  held out x = {x[held]}: line {b:.1f} + {w:.1f} x")
    oof[held, 0] = b + w * x[held]
    oof[held, 1] = [ys[np.argmin(abs(xs - t))] for t in x[held]]
w_ = np.linalg.solve(oof.T @ oof, oof.T @ y)
print("  out-of-fold SSE of A and B:", ((y[:, None] - oof) ** 2)
      .sum(0).round(2), " sum of weights", w_.sum().round(4))
print("  Z'(y - Zw) =", (oof.T @ (y - oof @ w_)).round(10))
xs = x; w = np.sum((xs - 3.5) * (y - y.mean())) / np.sum((xs - 3.5) ** 2)
print(f"  line on all six points: {y.mean() - w * 3.5:.1f} + {w:.1f} x")

print("Example 29.5-1")
P = np.array([[0.90, 0.80, 0.70], [0.55, 0.35, 0.40],
              [0.95, 0.30, 0.40], [0.20, 0.10, 0.30],
              [0.55, 0.60, 0.10]])
print("  3p^2 - 2p^3 at p = 0.7:", round(3 * 0.7 ** 2 - 2 * 0.7 ** 3, 4))
print("  member probabilities of patient 3 in order:", P[2])

print("Case study")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
est = [("lr", make_pipeline(sc(), LogisticRegression())),
       ("svc", make_pipeline(sc(), CalibratedClassifierCV(
           SVC(), method="sigmoid", cv=5, ensemble=False))),
       ("rf", RandomForestClassifier(100, random_state=0)),
       ("knn", make_pipeline(sc(), KNeighborsClassifier()))]
ll = [log_loss(yte, m.fit(Xtr, ytr).predict_proba(Xte))
      for _, m in est]
print("  mean of the four member log losses:", np.mean(ll).round(4))
knn = est[3][1]
pk = knn.predict_proba(Xte)[np.arange(len(yte)), yte]
n0 = int((pk == 0).sum())
print("  kNN patients with P(true class) = 0:", n0, "cost each",
      round(-np.log(1e-15) / len(yte), 4), "together",
      round(n0 * -np.log(1e-15) / len(yte), 4))
cv_lr = cross_val_score(est[0][1], Xtr, ytr, cv=5).mean()
cv_soft = cross_val_score(VotingClassifier(est, voting="soft"),
                          Xtr, ytr, cv=5).mean()
print("  5-fold gain of the soft vote over LR:", round(cv_soft - cv_lr, 4),
      "=", round((cv_soft - cv_lr) * len(ytr), 1), "of", len(ytr),
      "patients")
print("  standard error of an accuracy of 0.884 on 1000 cases:",
      round(float(np.sqrt(0.884 * 0.116 / 1000)), 4))
print("  standard error of 0.958 on 143 patients:",
      round(float(np.sqrt(0.958 * 0.042 / 143)), 4))
print("Condorcet: majority of 3 and 5 for p = 0.45:",
      round(float(majority(3, 0.45)), 4),
      round(float(majority(5, 0.45)), 4))

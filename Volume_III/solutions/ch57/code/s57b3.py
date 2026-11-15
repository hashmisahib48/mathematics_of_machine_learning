# Problem 57B.3: clipping the probabilities of a random forest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
ytr, yte = 1 - ytr, 1 - yte                  # 1 = malignant
rf = RandomForestClassifier(random_state=0).fit(Xtr, ytr)
p = rf.predict_proba(Xte)[:, 1]              # P(malignant)

for eps in (0, 0.005, 0.01, 0.02, 0.05):
    ll = log_loss(yte, np.clip(p, eps, 1 - eps))
    print(f"eps {eps:5.3f}  log loss {ll:.4f}")

sure = (p == 0) | (p == 1)                   # exactly 0 or 1
wrong = sure & ((p == 1) != (yte == 1))
print("forecasts exactly 0 or 1:", sure.sum(), " wrong:", wrong.sum())
for i in np.flatnonzero(wrong):
    print(f"test row {i}: forecast {p[i]:.1f}, truth {yte[i]}")
cost = -np.log(np.finfo(float).eps)      # charged by scikit-learn
print(f"-ln(2.2e-16) = {cost:.2f}; per patient {cost / len(yte):.4f}"
      f" of {log_loss(yte, p):.4f}")
rest = ~wrong
print("log loss of the other", rest.sum(), "patients:",
      round(log_loss(yte[rest], p[rest], labels=[0, 1]), 4))

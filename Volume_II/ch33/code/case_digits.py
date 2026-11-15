# Case study: ten digits with one-versus-rest and one-versus-one.
import json, time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier as OvO
from sklearn.multiclass import OneVsRestClassifier as OvR
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
n, K = len(ytr), 10

lr, svm, rbf = LogisticRegression(max_iter=300), LinearSVC(), SVC()
models = [("LogisticRegression (softmax)", lr, 1, n),
          ("OvR(LogisticRegression)", OvR(lr), K, K * n),
          ("OvO(LogisticRegression)", OvO(lr), 45, (K - 1) * n),
          ("LinearSVC (OvR inside)", svm, K, K * n),
          ("OvO(LinearSVC)", OvO(svm), 45, (K - 1) * n),
          ("SVC rbf (OvO inside)", rbf, 45, (K - 1) * n),
          ("OvR(SVC rbf)", OvR(rbf), K, K * n)]
print(f"{'model':30s}{'problems':>9s}{'rows':>8s}{'fit (s)':>9s}"
      f"{'test acc':>10s}")
out = []
for name, m, p, rows in models:
    t = []
    for _ in range(3):                      # best of three fits
        t0 = time.perf_counter()
        m.fit(Xtr, ytr)
        t.append(time.perf_counter() - t0)
    acc = m.score(Xte, yte)
    print(f"{name:30s}{p:9d}{rows:8d}{min(t):9.3f}{acc:10.4f}")
    out.append([name, p, rows, min(t), acc])
json.dump(out, open("verify/case_digits.json", "w"))

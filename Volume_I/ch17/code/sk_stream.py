# Learning from a stream with partial_fit (Section 17.10).
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

scaler = StandardScaler()
kw = dict(loss="log_loss", alpha=1e-3, random_state=0)
plain, avg = SGDClassifier(**kw), SGDClassifier(average=True, **kw)
rng = np.random.RandomState(0)
print("batch  rows seen   plain   average=True")
k = 0
for _ in range(2):                         # two passes
    for idx in np.array_split(rng.permutation(len(Xtr)), 6):
        scaler.partial_fit(Xtr[idx])       # running mean, variance
        Xb = scaler.transform(Xtr[idx])
        for m in (plain, avg):
            m.partial_fit(Xb, ytr[idx], classes=[0, 1])
        k += 1
        Xt = scaler.transform(Xte)
        print(f"{k:5d}  {k * 71:9d}   {plain.score(Xt, yte):.4f}"
              f"  {avg.score(Xt, yte):.4f}")

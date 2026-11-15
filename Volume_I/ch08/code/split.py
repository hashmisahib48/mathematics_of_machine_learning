# 8.1  train_test_split: a tiny table, the diabetes data, stratify.
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

X = np.arange(20).reshape(10, 2)       # ten samples, two features
y = np.arange(10) * 10                 # target of sample i is 10 i
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=0)
print("test rows :", y_te // 10)
print("train rows:", y_tr // 10)

n = len(load_diabetes().target)        # Eq. 8.1-1 for f = 0.25
a, b = train_test_split(np.arange(n), test_size=0.25,
                        random_state=0)
print("diabetes n =", n, " train", len(a), " test", len(b))
a, b = train_test_split(np.arange(n), test_size=0.25, shuffle=False)
print("shuffle=False: test rows start at", len(a))

X, y = load_breast_cancer(return_X_y=True)
print("whole data: class 1 share", round(y.mean(), 4))
for strat in (None, y):                # 200 different seeds each
    fr = [train_test_split(X, y, test_size=0.25, random_state=s,
                           stratify=strat)[3].mean()
          for s in range(200)]
    print("stratified" if strat is not None else "plain     ",
          "test share  min", round(min(fr), 3),
          " max", round(max(fr), 3), " sd", round(np.std(fr), 4))

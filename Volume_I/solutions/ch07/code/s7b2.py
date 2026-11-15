# Problem 7B.2: scatter of test accuracy over 200 splits (wine)
import numpy as np
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
acc = []
for seed in range(200):                      # 200 different splits
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression())
    acc.append(m.fit(Xtr, ytr).score(Xte, yte))
acc = np.array(acc)
m_test = len(yte)                            # 45 wines
print("test part:", m_test, "wines, one wine =", round(1 / m_test, 4))
print(f"mean {acc.mean():.4f}  sd {acc.std():.4f}")
print(f"min  {acc.min():.4f}  max {acc.max():.4f}")
a = acc.mean()
s = np.sqrt(a * (1 - a) / m_test)            # Eq. 7.9-1
print("binomial sd, Eq. 7.9-1:", round(float(s), 4))
vals, cnt = np.unique(acc.round(4), return_counts=True)
print("different values:", len(vals))
print("share of splits with at most 1 error:",
      round(float((acc >= 1 - 1.5 / m_test).mean()), 3))
print("splits at 1.0000:", int((acc == 1).sum()), "of 200")

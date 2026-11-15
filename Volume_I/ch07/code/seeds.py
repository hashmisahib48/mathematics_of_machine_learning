# random_state: same seed, same split; new seed, new score.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
a = train_test_split(X, y, test_size=0.25, random_state=7)
b = train_test_split(X, y, test_size=0.25, random_state=7)
c = train_test_split(X, y, test_size=0.25, random_state=8)
print("seed 7 twice identical :", all(map(np.array_equal, a, b)))
print("seed 7 and 8 identical :", all(map(np.array_equal, a, c)))
acc = []
for seed in range(200):                       # 200 different splits
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed)
    m = make_pipeline(StandardScaler(), LogisticRegression())
    acc.append(m.fit(Xtr, ytr).score(Xte, yte))
acc = np.array(acc)
print("test accuracy over 200 splits:")
print("  mean %.4f  sd %.4f" % (acc.mean(), acc.std()))
print("  min  %.4f  max %.4f" % (acc.min(), acc.max()))
a, m_test = acc.mean(), len(yte)              # 38 test flowers
print("binomial sd sqrt(a(1-a)/m):",
      round(float(np.sqrt(a * (1 - a) / m_test)), 4))
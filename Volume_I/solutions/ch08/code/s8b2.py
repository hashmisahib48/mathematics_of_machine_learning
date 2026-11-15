# Problem 8B.2: guessing in proportion (strategy="stratified")
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
y = (y == 9).astype(int)                        # 1 = a nine
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
p = np.bincount(ytr) / len(ytr)                 # training shares p_k
theory = (p ** 2).sum()                         # sum of p_k squared
print("shares p_k:", p.round(4), " sum p_k^2 =", round(theory, 4))

acc = np.array([DummyClassifier(strategy="stratified", random_state=s)
                .fit(Xtr, ytr).score(Xte, yte) for s in range(200)])
print(f"200 seeds: mean {acc.mean():.4f}  sd {acc.std():.4f}"
      f"  range {acc.min():.3f} to {acc.max():.3f}")

q = np.bincount(yte)                            # test samples/class
m = len(yte)
print("test part:", q.tolist(), " expected accuracy there:",
      round(float((q / m * p).sum()), 4))
# a sample of class k is guessed right with probability p_k
var = (q * p * (1 - p)).sum()                   # variance, count
sd = np.sqrt(var) / m
print("variance of the number right:", round(float(var), 2),
      " exact sd of the accuracy:", round(float(sd), 4))
a = theory                                      # Eq. 7.9-1
sb = np.sqrt(a * (1 - a) / m)
print("binomial sd sqrt(a(1-a)/m):", round(float(sb), 4))
se = acc.std() / np.sqrt(200)
print("standard error of the mean of 200:", round(float(se), 4))
ss = acc.std() / np.sqrt(398)                   # sd of an sd, n = 200
print("uncertainty of the sd:", round(float(ss), 4))

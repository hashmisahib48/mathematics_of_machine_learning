# Hand checks for Problems 26A.1 - 26A.4 and the arithmetic of 26C.1.  Run: python3 code/hand.py
import numpy as np

print("== 26A.1  five samples, stump wrong on samples 2 and 5")
w = np.full(5, 0.2); wrong = np.array([0, 1, 0, 0, 1], bool)
eps = w[wrong].sum(); alpha = np.log((1 - eps) / eps)
print(f"eps = {eps:.4f}   exp(alpha) = {np.exp(alpha):.4f}   alpha = {alpha:.4f}")
raw = w * np.exp(alpha * wrong); Z = raw.sum(); new = raw / Z
print("unnormalised:", raw.round(4).tolist(), " Z =", round(Z, 4))
print("new weights :", new.round(4).tolist(), " sum =", round(new.sum(), 4))
print("error of the same stump under the new weights:", round(new[wrong].sum(), 4))

print("\n== 26A.2  K = 3, six samples, three wrong, eps = 1/2")
for eta in (1.0, 0.5):
    a = eta * (np.log(0.5 / 0.5) + np.log(2)); f = np.exp(a)
    w = np.full(6, 1 / 6); wr = np.array([1, 1, 1, 0, 0, 0], bool)
    raw = w * np.exp(a * wr); Z = raw.sum(); n = raw / Z
    print(f"eta={eta}: alpha={a:.4f} factor={f:.4f} raw wrong={raw[0]:.4f} raw right={raw[3]:.4f} Z={Z:.4f}"
          f" -> {n[0]:.4f} {n[3]:.4f}  share of mistakes {n[wr].sum():.4f}")
print("random guessing on 3 classes is wrong with prob 1-1/K =", round(1 - 1 / 3, 4))

print("\n== 26A.3  a vote")
al = np.array([1.0, 0.6, 0.5]); h = np.array([1, -1, -1])
F = (al * h).sum(); f = 2 * F / al.sum(); P = 1 / (1 + np.exp(-f))
print(f"sum alpha h = {F:.4f}  sum alpha = {al.sum():.4f}  f = {f:.4f}  P(+1) = {P:.4f}  P(-1) = {1-P:.4f}")
print("largest possible |f| = 2 ->  P between", round(1/(1+np.exp(2)), 4), "and", round(1/(1+np.exp(-2)), 4))

print("\n== 26A.4  AdaBoost.R2, linear loss")
y = np.array([1, 2, 4, 8.]); pred = np.array([1, 1, 5, 5.]); w = np.full(4, 0.25)
e = np.abs(y - pred); D = e.max(); L = e / D
print("abs errors", e.tolist(), " D =", D, " L =", L.round(4).tolist())
eb = (w * L).sum(); beta = eb / (1 - eb); alpha = np.log(1 / beta)
print(f"eps_bar = {eb:.4f}  beta = {beta:.4f}  alpha = {alpha:.4f}")
fac = beta ** (1 - L); raw = w * fac; Z = raw.sum(); new = raw / Z
print("factors beta^(1-L):", fac.round(4).tolist())
print("w*factor          :", raw.round(4).tolist(), " sum =", round(Z, 4))
print("new weights       :", new.round(4).tolist(), " sum =", round(new.sum(), 4))
for name, Lx in (("square", L ** 2), ("exponential", 1 - np.exp(-L))):
    print(f"{name:11s} L = {Lx.round(4).tolist()}  eps_bar = {(w*Lx).sum():.4f}")

print("\n== 26C.1  alpha for eps = 0.2, 0.5, 0.8 (two classes) and 0.7 (three classes)")
for ep in (0.2, 0.5, 0.8):
    print(f"eps={ep}: alpha = {np.log((1-ep)/ep):+.4f}")
print(f"K=3, eps=0.7: alpha = {np.log(0.3/0.7)+np.log(2):+.4f}  (1-1/K = {1-1/3:.4f})")

print("\n== 26B.2  side remark: a fully grown tree depends on its random_state (ties between equally good splits)")
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
cd = load_breast_cancer()
Xtr, Xte, ytr, yte = train_test_split(cd.data, cd.target, test_size=0.25, random_state=0, stratify=cd.target)
print("test accuracy of DecisionTreeClassifier(random_state=s), s = 0..7:",
      [round(DecisionTreeClassifier(random_state=s).fit(Xtr, ytr).score(Xte, yte), 4) for s in range(8)])

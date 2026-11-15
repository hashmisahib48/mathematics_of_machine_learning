# VotingClassifier and StackingClassifier on noisy moons.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(1200, noise=0.35, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, train_size=200, random_state=0, stratify=y)
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
est = [("line", LogisticRegression()),
       ("near", KNeighborsClassifier(15)), ("tree", tree)]

for name, m in est:
    acc = m.fit(Xtr, ytr).score(Xte, yte)
    print(f"{name:5s} test accuracy {acc:.4f}")
hard = VotingClassifier(est).fit(Xtr, ytr)
soft = VotingClassifier(est, voting="soft").fit(Xtr, ytr)
wtd = VotingClassifier(est, voting="soft", weights=[1, 2, 1])
wtd.fit(Xtr, ytr)
for name, m in (("hard", hard), ("soft", soft), ("weighted", wtd)):
    print(f"{name:8s} vote  test accuracy {m.score(Xte, yte):.4f}")
print("classes_", soft.classes_, " members",
      list(soft.named_estimators_))
flat = soft.set_params(flatten_transform=False)
print("transform shapes: hard", hard.transform(Xte).shape,
      " soft", wtd.transform(Xte).shape,
      " soft, not flat", flat.transform(Xte).shape)
print("soft probabilities of test sample 0:",
      soft.predict_proba(Xte[:1]).round(3))

stack = StackingClassifier(est).fit(Xtr, ytr)
print("stack  test accuracy", round(stack.score(Xte, yte), 4))
print("stack_method_", stack.stack_method_)
fin = stack.final_estimator_
print("level-1 matrix", stack.transform(Xte).shape, " final",
      type(fin).__name__, " weights", fin.coef_.round(3),
      " intercept", fin.intercept_.round(3))

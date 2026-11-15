# AdaBoost.R2 on the diabetes data (the split of Chapter 9).
from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as DTR

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0)
tree = DTR(max_depth=3, random_state=0).fit(Xtr, ytr)
print(f"one tree, depth 3         test R2 {tree.score(Xte, yte):.3f}")
for loss in ("linear", "square", "exponential"):
    m = AdaBoostRegressor(loss=loss, random_state=0).fit(Xtr, ytr)
    print(f"AdaBoost.R2, {loss:11s}  test R2 {m.score(Xte, yte):.3f}"
          f"   rounds kept {len(m.estimators_)}")
m = AdaBoostRegressor(n_estimators=200, learning_rate=0.5,
                      random_state=0).fit(Xtr, ytr)
print(f"200 rounds, learning_rate 0.5: test R2 {m.score(Xte, yte):.3f}")

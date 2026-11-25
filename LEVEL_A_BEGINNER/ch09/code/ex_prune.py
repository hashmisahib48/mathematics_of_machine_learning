# Cost-complexity pruning of a tree for the breast cancer data.
from functools import partial
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

D = load_breast_cancer()
# a, b: training and test features; c, d: training and test classes
a, b, c, d = train_test_split(D.data, D.target, test_size=0.25,
                              random_state=0, stratify=D.target)
grow = partial(DecisionTreeClassifier, random_state=0)
path = grow(ccp_alpha=0.0).cost_complexity_pruning_path(a, c)
alphas = path.ccp_alphas[:-1]           # the last leaves only the root
cv = [cross_val_score(grow(ccp_alpha=al), a, c, cv=5).mean()
      for al in alphas]
best = alphas[int(np.argmax(cv))]
for al in (0.0, best):
    m = grow(ccp_alpha=al).fit(a, c)
    print(f"ccp_alpha {al:.4f}: leaves {m.get_n_leaves():2d},"
          f" test accuracy {m.score(b, d):.3f}")
imp = m.feature_importances_
for j in np.argsort(-imp)[:3]:
    print(f"  {D.feature_names[j]:22s} importance {imp[j]:.3f}")

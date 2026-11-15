# Case study: cost-complexity pruning, alpha by cross-validation.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0,
    stratify=data.target)
full = DecisionTreeClassifier(random_state=0)
path = full.cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]            # the last one is the root

scores = []
for a in alphas:
    m = DecisionTreeClassifier(ccp_alpha=a, random_state=0)
    scores.append(cross_val_score(m, X_tr, y_tr, cv=5).mean())
best = alphas[int(np.argmax(scores))]      # first maximum
print("best alpha", round(best, 4),
      " 5-fold", round(max(scores), 4))

pruned = DecisionTreeClassifier(ccp_alpha=best, random_state=0)
pruned.fit(X_tr, y_tr)
print("pruned tree: leaves", pruned.get_n_leaves(),
      " depth", pruned.get_depth(),
      " train", round(pruned.score(X_tr, y_tr), 4),
      " test", round(pruned.score(X_te, y_te), 4))
top = np.argsort(pruned.feature_importances_)[::-1][:3]
for j in top:
    print(f"{data.feature_names[j]:20s}",
          round(pruned.feature_importances_[j], 4))

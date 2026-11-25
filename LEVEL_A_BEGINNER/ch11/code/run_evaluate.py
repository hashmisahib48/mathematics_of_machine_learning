# The worked examples of Section 11.4 and plain cross-validation.
from evaluate import (StandardScaler, confusion, cross_val_score,
                      grid_search, precision_recall_f1, roc_auc)
from knn import KNeighborsClassifier
from sklearn.datasets import load_iris

train = [[10, 200], [12, 260], [14, 230], [16, 290]]
sc = StandardScaler().fit(train)
print("mean", sc.mean_, " scale", [round(s, 4) for s in sc.scale_])
print("test case (15, 300) ->",
      [round(v, 4) for v in sc.transform([[15, 300]])[0]])
y = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
s = [0.9, 0.8, 0.55, 0.3, 0.7, 0.4, 0.35, 0.2, 0.1, 0.05]
pred = [int(v >= 0.5) for v in s]
print("TP, FP, FN, TN:", confusion(y, pred))
print("precision, recall, F1:", precision_recall_f1(y, pred))
print("AUC", round(roc_auc(y, s), 4))
X, t = load_iris(return_X_y=True)
acc = cross_val_score(lambda: KNeighborsClassifier(5), X.tolist(),
                      t.tolist(), k=5)
print("5-fold accuracies", [round(a, 3) for a in acc],
      " mean", round(sum(acc) / 5, 3))
grid = {"n_neighbors": [1, 3, 5, 9, 15, 25]}
cv, best = grid_search(KNeighborsClassifier, X.tolist(),
                       t.tolist(), grid)
print("grid search: best", best, " CV accuracy", round(cv, 3))

# Problem 57B.1: ROC AUC of each single feature (breast cancer)
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score

data = load_breast_cancer()
X, names = data.data, data.feature_names
malignant = 1 - data.target          # in scikit-learn 0 = malignant

auc = np.array([roc_auc_score(malignant, X[:, j]) for j in range(30)])
folded = np.maximum(auc, 1 - auc)    # a score can be read either way
order = np.argsort(-folded)

print("five best features (AUC in the better direction)")
for j in order[:5]:
    print(f"  {names[j]:22s} AUC {auc[j]:.4f}")
print("features with folded AUC above 0.9:", (folded > 0.9).sum())
print("features with AUC below 0.5:", (auc < 0.5).sum())
for j in np.flatnonzero(auc < 0.5):
    m = X[malignant == 1, j].mean()
    b = X[malignant == 0, j].mean()
    print(f"  {names[j]:22s} AUC {auc[j]:.4f}")
    print(f"      mean malignant {m:.4f}, benign {b:.4f}")
k = np.argmin(np.abs(auc - 0.5))
print("weakest:", names[k], "AUC", round(auc[k], 4))
j = order[0]                         # reversing a score gives 1 - AUC
n1, n0 = malignant.sum(), (1 - malignant).sum()      # 212 and 357
se = np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
print(f"standard error of an AUC of 0.5: {se:.4f}")
rev = roc_auc_score(malignant, -X[:, j])
print(names[j], "reversed:", round(rev, 4))

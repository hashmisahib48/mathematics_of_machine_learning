# Problem 58B.2: spread of a permutation importance against n_repeats
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, 1 - data.target                 # 1 = malignant
Xa, Xb, ya, yb = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=data.target)
gb = GradientBoostingClassifier(random_state=0).fit(Xa, ya)
j = list(data.feature_names).index("worst texture")

sd = {}
for R in (1, 4, 16):
    v = [permutation_importance(
            gb, Xb, yb, scoring="neg_log_loss", n_repeats=R,
            random_state=seed).importances_mean[j]
         for seed in range(10)]
    sd[R] = np.std(v)
    print(f"R = {R:2d}: mean {np.mean(v):.4f}  sd over 10 seeds "
          f"{sd[R]:.4f}  1/sqrt(R) rule {sd[1] / np.sqrt(R):.4f}")
print("sd ratios: R = 1 to 4 %.2f (rule 2), 4 to 16 %.2f (rule 2)"
      % (sd[1] / sd[4], sd[4] / sd[16]))
print("sd ratio R = 1 to 16: %.2f (rule 4)" % (sd[1] / sd[16]))
one = permutation_importance(gb, Xb, yb, scoring="neg_log_loss",
                             n_repeats=16, random_state=0)
print("one run, R = 16: mean %.4f  sd over the shuffles %.4f"
      % (one.importances_mean[j], one.importances_std[j]))

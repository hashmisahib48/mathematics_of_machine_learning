# Case study, continued: the weights and an L1 penalty.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0,
    stratify=data.target)
names = data.feature_names

pipe = make_pipeline(StandardScaler(), LogisticRegression())
w = pipe.fit(X_tr, y_tr)[-1].coef_[0]   # per standard deviation
order = np.argsort(w)
print("most negative (towards malignant):")
for j in order[:4]:
    print(f"  {names[j]:22s}{w[j]:7.3f}  odds x {np.exp(w[j]):.2f}")
print("most positive (towards benign):")
for j in order[::-1][:4]:
    print(f"  {names[j]:22s}{w[j]:7.3f}  odds x {np.exp(w[j]):.2f}")

lasso = LogisticRegression(C=0.1, l1_ratio=1.0, solver="liblinear",
                           random_state=0)
sparse = make_pipeline(StandardScaler(), lasso).fit(X_tr, y_tr)
print("L1, C = 0.1: nonzero weights",
      int((sparse[-1].coef_ != 0).sum()),
      " test accuracy", round(sparse.score(X_te, y_te), 4))

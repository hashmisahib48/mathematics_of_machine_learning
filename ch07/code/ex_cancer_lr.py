# Logistic regression for the breast cancer data.
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = load_breast_cancer()                  # target 0 = malignant
a, b, c, d = train_test_split(D.data, D.target, test_size=0.25,
                              random_state=0, stratify=D.target)
m = make_pipeline(StandardScaler(), LogisticRegression()).fit(a, c)
print("test accuracy", round(m.score(b, d), 3))
print(confusion_matrix(d, m.predict(b)))
w = m[-1].coef_[0]
for j in np.argsort(-np.abs(w))[:4]:
    print(f"{D.feature_names[j]:22s} w {w[j]:6.2f}"
          f"  odds x {np.exp(w[j]):.2f}")

# How confident are the probabilities of naive Bayes?
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.25,
                              random_state=0, stratify=y)
lr = make_pipeline(StandardScaler(), LogisticRegression())
for name, m in [("naive Bayes", GaussianNB()), ("logistic", lr)]:
    p = m.fit(a, c).predict_proba(b)[:, 1]
    wrong = (p > 0.5) != (d == 1)
    sure = np.maximum(p, 1 - p) > 0.99      # confidence above 99 %
    print(f"{name:12s} errors {wrong.sum():2d}, made with over"
          f" 99 % confidence {(wrong & sure).sum()},"
          f" log loss {log_loss(d, p):.3f}")

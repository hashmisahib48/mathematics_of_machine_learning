# Case study: five tags per document, independent and chained.
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain as Chain
from sklearn.multioutput import MultiOutputClassifier

X, Y = make_multilabel_classification(
    n_samples=1500, n_features=20, n_classes=5, n_labels=2,
    length=50, allow_unlabeled=False, random_state=0)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3,
                                      random_state=0)
print("labels per document:", Y.sum(1).mean().round(2),
      " label frequencies:", Y.mean(0).round(2))

base = LogisticRegression(max_iter=1000)
models = {"independent": MultiOutputClassifier(base),
          "chain, column order": Chain(base),
          "chain, random order": Chain(base, order="random",
                                       random_state=0),
          "chain, cv=5": Chain(base, cv=5)}
print(f"{'':20s}{'Hamming':>8s}{'subset':>8s}{'F1 mic':>8s}"
      f"{'F1 mac':>8s}  5-fold subset")
P = np.zeros_like(Yte)                       # no label at all
print(f"{'no labels':20s}{hamming_loss(Yte, P):8.4f}"
      f"{accuracy_score(Yte, P):8.4f}{0:8.4f}{0:8.4f}")
for name, m in models.items():
    P = m.fit(Xtr, Ytr).predict(Xte)
    cv = cvs(m, X, Y, cv=5)
    print(f"{name:20s}{hamming_loss(Yte, P):8.4f}"
          f"{accuracy_score(Yte, P):8.4f}"
          f"{f1_score(Yte, P, average='micro'):8.4f}"
          f"{f1_score(Yte, P, average='macro'):8.4f}"
          f"  {cv.mean():.3f} +- {cv.std():.3f}")

# Case study: handwritten digits, 1347 training and 450 test images.
import warnings
import numpy as np
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")           # convergence notes
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler
def mlp(**kw): return MLPClassifier(random_state=0, **kw)
models = {
    "majority class": DummyClassifier(),
    "Gaussian naive Bayes": GaussianNB(),
    "5-NN, raw pixels": KNeighborsClassifier(),
    "5-NN, scaled": make_pipeline(sc(), KNeighborsClassifier()),
    "logistic regression, scaled":
        make_pipeline(sc(), LogisticRegression(max_iter=2000)),
    "MLP (100,), raw pixels": mlp(),
    "MLP (100,), scaled": make_pipeline(sc(), mlp()),
    "MLP scaled, alpha = 1": make_pipeline(sc(), mlp(alpha=1.0)),
    "MLP scaled, early stopping":
        make_pipeline(sc(), mlp(early_stopping=True)),
}
print(f"{'model':30s} train   test     cv")
for name, model in models.items():
    model.fit(Xtr, ytr)
    cv = cross_val_score(model, Xtr, ytr, cv=5, n_jobs=2).mean()
    print(f"{name:30s} {model.score(Xtr, ytr):.4f}  "
          f"{model.score(Xte, yte):.4f}  {cv:.4f}")

net = models["MLP (100,), scaled"][-1]
print("epochs", net.n_iter_, " parameters",
      sum(w.size for w in net.coefs_ + net.intercepts_))
test = [make_pipeline(sc(), mlp()).set_params(
        mlpclassifier__random_state=s).fit(Xtr, ytr).score(Xte, yte)
        for s in range(10)]
print(f"ten seeds, test: {min(test):.4f} to {max(test):.4f}"
      f", mean {np.mean(test):.4f}")

import warnings
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline

warnings.simplefilter("ignore")
X, y = load_digits(return_X_y=True)
X = X / 16.0                              # pixels lie in [0, 1]
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)

raw = LogisticRegression(C=100, max_iter=3000).fit(Xtr, ytr)
rbm = BernoulliRBM(n_components=100, learning_rate=0.1, n_iter=30,
                   random_state=0)
pipe = make_pipeline(rbm, LogisticRegression(C=100, max_iter=3000))
pipe.fit(Xtr, ytr)
print(f"logistic on pixels   train {raw.score(Xtr, ytr):.4f}"
      f"  test {raw.score(Xte, yte):.4f}")
print(f"RBM + logistic       train {pipe.score(Xtr, ytr):.4f}"
      f"  test {pipe.score(Xte, yte):.4f}")

H = rbm.transform(Xte)                    # Eq. 31.11-3
print("features:", H.shape, " mean activity", H.mean().round(3))
print("components_", rbm.components_.shape)
print("intercept_visible_", rbm.intercept_visible_.shape,
      " intercept_hidden_", rbm.intercept_hidden_.shape)
pl_tr = rbm.score_samples(Xtr > 0.5).mean()   # Eq. 31.11-8
pl_te = rbm.score_samples(Xte > 0.5).mean()
print(f"pseudo-likelihood  train {pl_tr:.2f}  test {pl_te:.2f}")

# RandomTreesEmbedding: leaf indicators for a linear model.
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X, y = make_circles(n_samples=400, noise=0.1, factor=0.4,
                    random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.5, random_state=0, stratify=y)

raw = LogisticRegression().fit(X_tr, y_tr)
print("logistic regression on (x1, x2): test",
      raw.score(X_te, y_te))

emb = RandomTreesEmbedding(n_estimators=20, max_depth=3,
                           random_state=0)
Z_tr = emb.fit_transform(X_tr)
print("embedding:", Z_tr.shape, " ones per row:",
      int(Z_tr[0].sum()), "= number of trees")
print("leaves per tree:",
      [int(t.get_n_leaves()) for t in emb.estimators_[:4]])
pipe = make_pipeline(emb, LogisticRegression(max_iter=1000))
pipe.fit(X_tr, y_tr)
print("embedding + logistic regression: test",
      pipe.score(X_te, y_te))

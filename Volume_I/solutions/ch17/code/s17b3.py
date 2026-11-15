# Problem 17B.3: one-versus-rest SGD on the standardised iris data.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
m = SGDClassifier(random_state=0, max_iter=1000).fit(X, y)

print("coef_ shape:", m.coef_.shape,
      " intercept_ shape:", m.intercept_.shape)
F = X @ m.coef_.T + m.intercept_          # one score per class
print("scores of samples 0, 50, 100:\n", F[[0, 50, 100]].round(3))
pred = m.classes_[F.argmax(axis=1)]       # largest score wins
print("argmax agrees with predict:",
      np.array_equal(pred, m.predict(X)))
print("training accuracy:", round(m.score(X, y), 4))
for k in range(3):                        # class k against the rest
    s = np.where(y == k, 1, -1)
    print(f"class {k}: row accuracy against the rest",
          round(np.mean(np.sign(F[:, k]) == s), 3))
print("samples with no positive score:",
      int((F.max(axis=1) <= 0).sum()))

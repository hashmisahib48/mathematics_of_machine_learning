# SVC on the eight seeds: attributes and their sign conventions.
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

X = np.array([[1, 5], [2, 3], [3, 4], [4, 1],
              [7, 6], [8, 2], [4, 7], [6, 3]], float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])        # labels 0 and 1

lin = SVC(kernel="linear", C=10).fit(X, y)
print("classes_       :", lin.classes_)
print("support_ (0-based):", lin.support_,
      " n_support_:", lin.n_support_)
print("support_vectors_ :", lin.support_vectors_.tolist())
print("dual_coef_       :", lin.dual_coef_)
print("coef_, intercept_:", lin.coef_, lin.intercept_)
f = X @ lin.coef_[0] + lin.intercept_[0]
print("decision_function = Xw + b:",
      np.allclose(f, lin.decision_function(X)))
print("predict = classes_[f > 0]:",
      np.array_equal(lin.classes_[(f > 0) * 1], lin.predict(X)))

rbf = SVC(C=10).fit(X, y)             # kernel="rbf", "scale"
g = 1 / (X.shape[1] * X.var())
print("\ngamma='scale' = 1/(p Var(X)) =", round(g, 5))
K = rbf_kernel(X, rbf.support_vectors_, gamma=g)
f = K @ rbf.dual_coef_[0] + rbf.intercept_[0]  # Eq. 22.5-1
print("Eq. 22.5-1 reproduces decision_function:",
      np.allclose(f, rbf.decision_function(X)))
print("n_support_:", rbf.n_support_, " n_iter_:", rbf.n_iter_)
print("train accuracy:", rbf.score(X, y))
try:
    rbf.predict_proba(X)
except AttributeError as e:
    print("predict_proba ->", e)

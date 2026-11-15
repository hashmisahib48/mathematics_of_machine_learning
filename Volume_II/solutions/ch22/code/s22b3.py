# Problem 22B.3: a kernel handed to SVC as a Gram matrix.
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
g = 0.5
a = SVC(kernel="rbf", gamma=g).fit(X_tr, y_tr)
K_tr = rbf_kernel(X_tr, X_tr, gamma=g)    # 150 x 150
b = SVC(kernel="precomputed").fit(K_tr, y_tr)
K_te = rbf_kernel(X_te, X_tr, gamma=g)    # 50 x 150
fa, fb = a.decision_function(X_te), b.decision_function(K_te)
print("Gram matrices:", K_tr.shape, K_te.shape)
print("max |f_rbf - f_pre| =", f"{np.abs(fa - fb).max():.1e}")
same = np.array_equal(a.support_, b.support_)
print("same support vectors:", same, len(a.support_))
dc = np.abs(a.dual_coef_ - b.dual_coef_).max()
print("max |dual_coef_ diff| =", f"{dc:.1e}")
print("intercepts:", a.intercept_, b.intercept_)
print("accuracy:", a.score(X_te, y_te), b.score(K_te, y_te))
same = np.array_equal(a.predict(X_te), b.predict(K_te))
print("same predictions:", same)
try:
    b.decision_function(X_te)    # features, not kernel rows
except ValueError as e:
    print("wrong input:", str(e)[:42])

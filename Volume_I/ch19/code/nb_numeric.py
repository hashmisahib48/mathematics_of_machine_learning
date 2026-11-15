import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB

X = np.array([[1, 2], [2, 4], [3, 3], [4, 5], [6, 9], [8, 7]])
y = np.array([0, 0, 0, 1, 1, 1])
g = GaussianNB().fit(X, y)
print("theta_ =", g.theta_.round(4).tolist())
print("var_   =", g.var_.round(4).tolist())
print("class_prior_ =", g.class_prior_,
      " epsilon_ =", f"{g.epsilon_:.3e}")
q = np.array([[3.0, 6.0]])
print("predict_proba(q) =", g.predict_proba(q).round(4))

# by hand, Eq. 19.4-3
v = g.var_
ll = -0.5 * (np.log(2 * np.pi * v) + (q - g.theta_)**2 / v)
hand = np.log(g.class_prior_) + ll.sum(axis=1)
print("hand log joint =", hand.round(4))
print("predict_joint_log_proba =",
      g.predict_joint_log_proba(q).round(4))

# two batches with partial_fit give the same fit
h = GaussianNB()
h.partial_fit(X[[0, 1, 3, 4]], y[[0, 1, 3, 4]], classes=[0, 1])
h.partial_fit(X[[2, 5]], y[[2, 5]])
print("partial_fit equal:", np.allclose(h.var_, g.var_),
      np.allclose(h.theta_, g.theta_))

# categorical features: outlook (0, 1, 2) and wind (0, 1)
Xc = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]])
yc = np.array([1, 0, 1, 1, 1, 0])
cat = CategoricalNB().fit(Xc, yc)
print("category_count_[0]:", cat.category_count_[0].tolist())
cat3 = CategoricalNB(min_categories=[4, 2]).fit(Xc, yc)
print("unseen outlook 3:", cat3.predict_proba([[3, 1]]).round(4))

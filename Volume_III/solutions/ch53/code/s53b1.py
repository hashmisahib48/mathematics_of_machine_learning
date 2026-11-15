# Problem 53B.1: F and mutual information on a curved relation
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

rng = np.random.RandomState(1)
x = rng.uniform(-1, 1, 300)         # the informative column
noise = rng.uniform(-1, 1, 300)     # an unrelated column
y = (np.abs(x) > 0.5).astype(int)   # class 1 in the outer half
X = np.c_[x, noise]

F, p = f_classif(X, y)
mi = mutual_info_classif(X, y, random_state=0)
m0, m1 = x[y == 0], x[y == 1]
print("class sizes     :", np.bincount(y))
print("class means of x: %.4f  %.4f" % (m0.mean(), m1.mean()))
print("class sds of x  : %.4f  %.4f" % (m0.std(), m1.std()))
print("F               : x %.4f   noise %.4f" % tuple(F))
print("p-value         : x %.4f   noise %.4f" % tuple(p))
print("mutual info     : x %.4f   noise %.4f" % tuple(mi))

# the largest possible mutual information is H(y)
q = y.mean()
H = -(q * np.log(q) + (1 - q) * np.log(1 - q))
print("entropy of y    : %.4f nats" % H)

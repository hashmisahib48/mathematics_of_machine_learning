# SVC with three classes: one-vs-one inside, one-vs-rest on top.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
Z = StandardScaler().fit_transform(X)
m = SVC(kernel="rbf", C=1.0).fit(Z, y)
print("n_support_:", m.n_support_)
print("dual_coef_ shape:", m.dual_coef_.shape,
      " intercept_ shape:", m.intercept_.shape)

ovo = SVC(decision_function_shape="ovo").fit(Z, y)
d = ovo.decision_function(Z)                 # one column per pair
print("ovo shape:", d.shape, " ovr shape:",
      m.decision_function(Z).shape)
x = 100                                      # one flower, by hand
pairs = [(0, 1), (0, 2), (1, 2)]
votes, conf = np.zeros(3), np.zeros(3)
for k, (i, j) in enumerate(pairs):
    win = j if d[x, k] < 0 else i            # > 0: first class
    votes[win] += 1
    conf[i] += d[x, k]                       # confidence of pair k
    conf[j] -= d[x, k]
print("pair scores :", np.round(d[x], 3))
print("votes       :", votes)
print("sum of conf :", np.round(conf, 3))
f = votes + conf / (3 * (np.abs(conf) + 1))
print("votes + c/(3(|c|+1)):", np.round(f, 4))
print("ovr decision_function:",
      np.round(m.decision_function(Z)[x], 4))

pred = m.predict(Z)
ovr = m.decision_function(Z)
print("predict == argmax of ovr:", np.mean(pred == ovr.argmax(1)))
print("training accuracy:", round(m.score(Z, y), 4))

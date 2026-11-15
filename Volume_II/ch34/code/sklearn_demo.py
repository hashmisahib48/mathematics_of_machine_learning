# The chain and the 1-D set of Section 34.1 in scikit-learn.
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import (LabelPropagation,
    LabelSpreading, SelfTrainingClassifier)
np.set_printoptions(precision=4, suppress=True)

W = np.zeros((5, 5))                 # the chain of Example 34.5-1
for i, w in enumerate([2, 4, 1, 1]):
    W[i, i + 1] = W[i + 1, i] = w
X = np.arange(5.0).reshape(-1, 1)    # placeholder features
y = np.array([0, -1, -1, -1, 1])     # -1 marks unlabelled samples
chain = lambda A, B: W.copy()        # a kernel may be a function

lp = LabelPropagation(kernel=chain, max_iter=10000, tol=1e-12)
lp.fit(X, y)
print("LP n_iter_ =", lp.n_iter_, " labels =", lp.transduction_)
print("LP P(class 1) =", lp.label_distributions_[:, 1])
ls = LabelSpreading(kernel=chain, alpha=0.5, max_iter=10000,
                    tol=1e-12).fit(X, y)
print("LS n_iter_ =", ls.n_iter_, " labels =", ls.transduction_)
print("LS P(class 1) =", ls.label_distributions_[:, 1])
dflt = LabelPropagation(kernel=chain).fit(X, y)     # tol = 1e-3
print("default tol: n_iter_ =", dflt.n_iter_, " P(class 1) =",
      dflt.label_distributions_[:, 1])

x = np.array([0, 1, 2, 3.4, 4.2, 5, 6, 7]).reshape(-1, 1)
z = np.array([0, 0, -1, -1, -1, -1, 1, 1])
base = KNeighborsClassifier(n_neighbors=2)
st = SelfTrainingClassifier(base, threshold=0.75).fit(x, z)
print("\nlabeled_iter_ :", st.labeled_iter_)
print("transduction_ :", st.transduction_)
print("n_iter_ =", st.n_iter_, " termination_condition_ =",
      st.termination_condition_)
print("predict_proba at 3.4 :", st.predict_proba([[3.4]]))

import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (SimpleImputer, MissingIndicator,
                            KNNImputer, IterativeImputer)
from clinic import X
np.set_printoptions(precision=2, suppress=True, linewidth=70)

si = SimpleImputer(strategy="mean").fit(X)
print("statistics_ ", si.statistics_)
mi = MissingIndicator().fit(X)               # features="missing-only"
print("indicator   ", mi.features_, mi.transform(X).sum(axis=0))
Z = SimpleImputer(add_indicator=True).fit_transform(X)
print("add_indicator shape", Z.shape)
kn = KNNImputer(n_neighbors=2).fit(X)
print("KNN k=2, patient 4:", kn.transform(X)[3])
it = IterativeImputer(random_state=0).fit(X)
print("iterative, patient 4:", it.transform(X)[3], "n_iter_", it.n_iter_)
new = np.array([[52, np.nan, 98, 27.]])      # a new patient
print("new patient, KNN:", kn.transform(new)[0])

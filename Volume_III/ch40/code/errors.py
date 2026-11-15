# Messages quoted in the COMMON ERROR box and the parameter notes.
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture
from people import people_2d

X, _ = people_2d()
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    GaussianMixture(3, max_iter=3, random_state=0).fit(X)
print("1.", w[0].category.__name__, "-", str(w[0].message))

dup = np.vstack([np.zeros((4, 2)), X[:30]])           # four identical rows
try:
    GaussianMixture(6, reg_covar=0.0, init_params="random_from_data",
                    random_state=1).fit(dup)
    print("2. no error")
except ValueError as e:
    print("2. ValueError -", str(e))

try:
    GaussianMixture(3).fit(X[:2])
except ValueError as e:
    print("3. ValueError -", str(e))

gm = GaussianMixture(3, random_state=0).fit(X)
print("4. component labels depend on the seed:")
for s in (0, 1, 2):
    g2 = GaussianMixture(3, random_state=s).fit(X)
    print("   seed", s, " means of height (cm)", g2.means_[:, 0].round(0))

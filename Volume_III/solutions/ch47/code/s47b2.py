# Problem 47B.2: raw or standardised petals (held-out ln f).
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

R = load_iris().data[:, 2:4]        # petal length, width (cm)
sd = R.std(axis=0)
Z = (R - R.mean(axis=0)) / sd       # standardised copy
outer = KFold(5, shuffle=True, random_state=0)
inner = KFold(5, shuffle=True, random_state=1)


def held_out(D, grid):
    """Mean held-out ln f; h from an inner 5-fold search."""
    scores, hs = [], []
    for tr, te in outer.split(D):
        gs = GridSearchCV(KernelDensity(), {"bandwidth": grid},
                          cv=inner).fit(D[tr])
        scores.append(gs.score(D[te]) / len(te))
        hs.append(gs.best_params_["bandwidth"])
    return np.mean(scores), np.round(hs, 2)


ll_z, h_z = held_out(Z, np.arange(0.03, 0.50, 0.01))
ll_r, h_r = held_out(R, np.arange(0.03, 1.00, 0.01))
jac = np.log(sd).sum()         # ln f_x = ln f_z - sum ln sd_j
print("standardised: ln f", round(ll_z, 3), " h per fold", h_z)
print("raw         : ln f", round(ll_r, 3), " h per fold", h_r)
print("sd of columns", sd.round(3), " sum ln sd", round(jac, 3))
print("raw score in standardised units:", round(ll_r + jac, 3))
print("gain from standardising:", round(ll_z - (ll_r + jac), 3))
print("wrong sign of the Jacobian:", round(ll_r - jac, 3))

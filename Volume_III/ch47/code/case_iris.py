# Iris petals: kernel density against mixtures (held-out ln f).
import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(load_iris().data[:, 2:4])
outer = KFold(5, shuffle=True, random_state=0)
inner = KFold(5, shuffle=True, random_state=1)
grid = {"bandwidth": np.arange(0.03, 0.50, 0.01)}
mix = dict(n_init=3, reg_covar=1e-3, random_state=0)
models = {"KDE, h by CV": GridSearchCV(KernelDensity(), grid, cv=inner),
          "KDE, Scott": KernelDensity(bandwidth="scott"),
          "mixture K=2": GaussianMixture(2, **mix),
          "mixture K=3": GaussianMixture(3, **mix)}

mean_ll = lambda est, X: est.score_samples(X).mean()

for name, m in models.items():
    s = cross_val_score(m, Z, cv=outer, scoring=mean_ll)
    print(f"{name:13s} {s.mean():7.3f}   folds {s.round(2)}")

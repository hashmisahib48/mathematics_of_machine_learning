# Graphical lasso on 150 days of standardised returns.
import warnings
import numpy as np
from returns import true_model, draw
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

Sigma, Theta, sd = true_model()
Z = StandardScaler().fit_transform(draw(150, seed=1))
edge = lambda P: (np.abs(P) > 1e-10) & ~np.eye(len(P), dtype=bool)
true = edge(Theta)
print("true edges", true.sum() // 2)

print("alpha  edges  true  false  n_iter")
for a in (0.1, 0.2, 0.3, 0.4, 0.6):
    g = GraphicalLasso(alpha=a, enet_tol=1e-6).fit(Z)
    E = edge(g.precision_)
    print(f"{a:5.2f}{E.sum() // 2:7d}{(E & true).sum() // 2:6d}"
          f"{(E & ~true).sum() // 2:7d}{g.n_iter_:8d}")

with warnings.catch_warnings(record=True) as w:   # default tol
    warnings.simplefilter("always")
    g = GraphicalLasso(alpha=0.3).fit(Z)
print("alpha 0.3, enet_tol 1e-4: n_iter", g.n_iter_,
      "warnings:", [x.category.__name__ for x in w])

cv = GraphicalLassoCV(enet_tol=1e-6).fit(Z)
r = cv.cv_results_
k = np.argmax(r["mean_test_score"])
print("GraphicalLassoCV: alpha_", round(cv.alpha_, 4),
      " grid points", len(r["alphas"]))
print("best mean score", r["mean_test_score"][k].round(3),
      "; at alpha 0:", r["mean_test_score"][-1].round(3))

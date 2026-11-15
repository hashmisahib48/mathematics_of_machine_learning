import numpy as np
from sklearn.mixture import GaussianMixture

x = np.array([154., 160., 164., 170., 176., 184.]).reshape(-1, 1)
start = dict(weights_init=[0.5, 0.5], means_init=[[158.], [178.]],
             precisions_init=[[[1 / 64]], [[1 / 64]]])
gm = GaussianMixture(2, tol=1e-8, max_iter=500, random_state=0,
                     **start).fit(x)
print("converged", gm.converged_, " iterations", gm.n_iter_)
print("weights_ ", gm.weights_.round(4))
print("means_   ", gm.means_.ravel().round(3))
print("variances", gm.covariances_.ravel().round(3))
print("mean log-likelihood", round(gm.score(x), 4))
print("BIC", round(gm.bic(x), 3), " AIC", round(gm.aic(x), 3))

new = np.array([[170.]])
print("predict_proba(170)", gm.predict_proba(new).round(4))
print("log density(170)  ", gm.score_samples(new).round(4))
print("hard label(170)   ", gm.predict(new))
draw, comp = gm.sample(4)             # new heights from the model
print("sample", draw.ravel().round(1), "from components", comp)

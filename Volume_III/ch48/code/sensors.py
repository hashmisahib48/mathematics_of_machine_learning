# Invented pump data: temperature (deg C) and vibration (mm/s).
import numpy as np

MODES = [((55.0, 1.0), (3.0, 0.25), 0.5),     # idle: mean, sd, correlation
         ((75.0, 2.5), (4.0, 0.50), 0.6)]     # under load

def make_sensors(n=300, n_fault=15, seed=0):
    """n normal readings from two operating modes + n_fault faults.
    Returns X (n+n_fault, 2) and y (1 = fault, 0 = normal)."""
    rng = np.random.RandomState(seed)
    mode = rng.rand(n) < 0.5
    X = np.empty((n, 2))
    for k, (mu, sd, rho) in enumerate(MODES):
        cov = np.array([[sd[0]**2, rho*sd[0]*sd[1]],
                        [rho*sd[0]*sd[1], sd[1]**2]])
        m = mode == bool(k)
        X[m] = rng.multivariate_normal(mu, cov, size=m.sum())
    F = []
    while len(F) < n_fault:                    # faults: not near either mode
        q = rng.uniform([42, 0.0], [90, 5.0])
        d = [np.hypot((q[0]-mu[0])/sd[0], (q[1]-mu[1])/sd[1])
             for mu, sd, _ in MODES]
        if min(d) > 3.2:
            F.append(q)
    return np.vstack([X, F]), np.r_[np.zeros(n), np.ones(n_fault)]

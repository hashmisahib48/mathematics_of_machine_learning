# Facts about losses, schedules and the mistake bound, checked against
# scikit-learn 1.9.1.  Run: python3 fig/checks.py
# Writes verify/checks.json (numbers quoted in the text).
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE + '/..')
import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.linear_model._sgd_fast import Hinge, SquaredHinge, ModifiedHuber
R = {}

# 1. loss values at selected margins (library objects where they are exposed)
zs = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
tab = {}
for name, obj in [('hinge', Hinge(1.0)), ('perceptron', Hinge(0.0)),
                  ('squared_hinge', SquaredHinge(1.0)), ('modified_huber', ModifiedHuber())]:
    tab[name] = [obj.py_loss(z, 1.0) for z in zs]
tab['log_loss'] = [float(np.log1p(np.exp(-z))) for z in zs]
tab['zero_one'] = [float(z <= 0) for z in zs]
R['loss_at_margin'] = {'z': zs, **{k: [round(v, 4) for v in vs] for k, vs in tab.items()}}

# 2. one SGD step from weight w0 on a single sample x = 1, y = +1 with eta = 1, no penalty:
#    the new weight is w0 - L'(f).  The second sample has weight 0 (it only declares the classes).
def step(loss, w0, **kw):
    m = SGDClassifier(loss=loss, penalty=None, learning_rate='constant', eta0=1.0, fit_intercept=False,
                      shuffle=False, max_iter=1, tol=None, **kw)
    m.fit(np.array([[1.0], [-1.0]]), np.array([1, 0]), coef_init=np.array([[w0]]),
          intercept_init=np.array([0.0]), sample_weight=np.array([1.0, 0.0]))
    return float(m.coef_[0, 0])
losses = ['hinge', 'perceptron', 'squared_hinge', 'modified_huber', 'log_loss', 'squared_error']
R['classifier_step'] = {l: {f'w0={w0:g}': round(step(l, w0), 4) for w0 in (0.0, -5.0, 2.0)} for l in losses}

# 3. first 'optimal' rate for every loss: coefficient after one update / (-slope at z = 0)
alpha = 1e-4; slope0 = {'hinge': 1, 'perceptron': 1, 'squared_hinge': 2, 'modified_huber': 2, 'log_loss': 0.5, 'squared_error': 1}
R['optimal_eta1'] = {}
for l in losses:
    m = SGDClassifier(loss=l, penalty=None, learning_rate='optimal', alpha=alpha, fit_intercept=False,
                      shuffle=False, max_iter=1, tol=None)
    m.fit(np.array([[1.0], [-1.0]]), np.array([1, 0]), sample_weight=np.array([1.0, 0.0]))
    R['optimal_eta1'][l] = round(float(m.coef_[0, 0]) / slope0[l], 4)
R['optimal_eta1']['alpha^-1/4'] = round(alpha ** -0.25, 4)

# 4. regression losses: one step from w = 0, eta = 1, x = 1, for targets 0.5 and 3
def rstep(loss, yv, eps):
    m = SGDRegressor(loss=loss, epsilon=eps, penalty=None, learning_rate='constant', eta0=1.0, fit_intercept=False,
                     shuffle=False, max_iter=1, tol=None)
    m.partial_fit(np.array([[1.0]]), np.array([yv]))
    return round(float(m.coef_[0]), 4)
R['regressor_step_eps1'] = {l: {'y=0.5': rstep(l, 0.5, 1.0), 'y=3': rstep(l, 3.0, 1.0)}
                            for l in ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']}

# 5. mistake bound for the four tiles (Section 17.12)
X = np.array([[4, 3], [1, 0], [3, 4], [0, 1]], float); y = np.array([1, -1, 1, -1])
Xa = np.c_[X, np.ones(4)]                       # augmented with a 1 for the intercept
ws = np.array([1.0, 1.0, -4.0])                 # a separating vector found by hand
marg = y * (Xa @ ws) / np.linalg.norm(ws)
Rr = np.linalg.norm(Xa, axis=1).max()
R['bound'] = {'w_star': ws.tolist(), 'gamma': round(float(marg.min()), 4), 'gamma_sq': round(float(marg.min()) ** 2, 4),
              'R': round(float(Rr), 4), 'R_sq': round(float(Rr ** 2), 4),
              'bound': round(float(Rr ** 2 / marg.min() ** 2), 2), 'mistakes_actual': 5}
json.dump(R, open('verify/checks.json', 'w'), indent=1); print(json.dumps(R))

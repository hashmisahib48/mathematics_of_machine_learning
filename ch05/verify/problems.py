import sys; sys.path.insert(0, '../code')
import math
import numpy as np, statsmodels.api as sm
from gd import LinearRegressionGD
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
s = sm.datasets.stackloss.load_pandas().data; X = s[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; y = s.STACKLOSS.values
for name, D in [('raw', X), ('std', (X - X.mean(0)) / X.std(0))]:
    Dc = D - D.mean(0); A = Dc.T @ Dc; n = len(y); eta = 0.9 * n / np.trace(A); lam = np.linalg.eigvalsh(A)
    m = LinearRegressionGD(eta=eta, n_iter=200000, tol=1e-6).fit(D.tolist(), y.tolist())
    print('5-1', name, 'eta', round(eta, 5), 'iters', m.n_iter_, 'w', np.round(m.coef_, 4), 'b', round(m.intercept_, 3), 'cond', round(lam.max() / lam.min(), 1))
print('5-1 LS', LinearRegression().fit(X, y).coef_.round(4))
X, y = load_diabetes(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
for eta0 in (0.001, 0.01, 0.1):
    m = make_pipeline(StandardScaler(), SGDRegressor(eta0=eta0, max_iter=1000, tol=1e-6, penalty=None, random_state=0)).fit(a, c)
    print('5-2 eta0', eta0, 'epochs', m[-1].n_iter_, 'test R2', round(m.score(b, d), 3))
L = sm.datasets.longley.load_pandas().data; cols = ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']
Z = StandardScaler().fit_transform(L[cols].values); yy = L.TOTEMP.values; lam = np.linalg.eigvalsh(Z.T @ Z)
print('5-3 cond', '%.3g' % (lam.max() / lam.min()), 'factor per iter at eta=n/lmax*0.9', 1 - 2 * 0.9 * lam.min() / lam.max())
rho3 = 1 - 2 * 0.9 * lam.min() / lam.max(); print('5-3c rho', round(rho3, 5), 'iterations for factor 1000', math.ceil(math.log(1000) / -math.log(rho3)))
for it in (1000, 10000, 100000):
    m = LinearRegressionGD(eta=0.9 * 16 / lam.max(), n_iter=it, tol=0).fit(Z.tolist(), yy.tolist())
    print('5-3 iters', it, 'w', np.round(m.coef_, 0))
print('5-3 LS', LinearRegression().fit(Z, yy).coef_.round(0))
e = sm.datasets.engel.load_pandas().data; lx = np.log(e.income.values); ly = np.log(e.foodexp.values); n = len(lx)
xt = lx - lx.mean(); Sxx = xt @ xt; print('5-4 n', n, 'Sxx', round(Sxx, 4), 'limit', round(n / Sxx, 4), 'best', round(n / (2 * Sxx), 4))
for eta in (0.5 * n / (2 * Sxx), n / (2 * Sxx), 1.5 * n / (2 * Sxx)):
    m = LinearRegressionGD(eta=eta, n_iter=100000, tol=1e-10).fit(lx[:, None].tolist(), ly.tolist()); print('5-4 eta', round(eta, 4), 'iters', m.n_iter_, 'w', round(m.coef_[0], 4))
# 5-1d hand iterations, stack loss vs air flow
import math
n, Sxx, Sxy = 21, 1681.142857, 1715.285714; ws = Sxy / Sxx; eta = 0.01; rho = 1 - 2 * eta * Sxx / n
print('5-1d w*', round(ws, 4), 'limit', round(n / Sxx, 6), 'best', round(n / (2 * Sxx), 6), 'rho', round(rho, 5))
w = 0.0
for k in range(1, 4):
    e_sum = Sxy - w * Sxx; w = w + 2 * eta / n * e_sum; print('5-1d k', k, 'w', round(w, 4), 'w-w*', round(w - ws, 4))
print('5-1d iterations for |err|<0.001:', math.ceil(math.log(0.001 / ws) / math.log(abs(rho))))

Sxx_e = xt @ xt; yt = ly - ly.mean(); Sxy_e = xt @ yt; we = Sxy_e / Sxx_e; be = ly.mean() - we * lx.mean()
print('5-4c Sxy', round(Sxy_e, 4), 'Sxx', round(Sxx_e, 4), 'w*', round(we, 4), 'mean lx', round(lx.mean(), 4), 'mean ly', round(ly.mean(), 4), 'b', round(be, 4), 'pred 1000', round(math.exp(be + we * math.log(1000)), 1))
import math; print('5-5d updates', [math.ceil(331 / m) for m in (1, 8, 32, 331)], 'work per epoch 331 each')

# added parts
from sklearn.linear_model import Ridge
s = sm.datasets.stackloss.load_pandas().data; X = s[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; y = s.STACKLOSS.values
new = np.array([[60., 20., 85.]]); mu, sd = X.mean(0), X.std(0)
Dr = LinearRegressionGD(eta=0.9 * 21 / np.trace((X - mu).T @ (X - mu)), n_iter=200000, tol=1e-6).fit(X.tolist(), y.tolist())
Zs = (X - mu) / sd; Ds = LinearRegressionGD(eta=0.9 * 21 / np.trace(Zs.T @ Zs), n_iter=200000, tol=1e-6).fit(Zs.tolist(), y.tolist())
print('5-1f raw', round(Dr.predict(new.tolist())[0], 3), 'std', round(Ds.predict(((new - mu) / sd).tolist())[0], 3), 'LS', round(LinearRegression().fit(X, y).predict(new)[0], 3))
X, y = load_diabetes(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
sg = make_pipeline(StandardScaler(), SGDRegressor(eta0=0.01, max_iter=1000, tol=1e-6, penalty='l2', alpha=1.0, random_state=0)).fit(a, c)
rg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(a, c)
print('5-2d sgd l2 w', np.round(sg[-1].coef_[:3], 2), 'R2', round(sg.score(b, d), 3), '| ridge w', np.round(rg[-1].coef_[:3], 2), 'R2', round(rg.score(b, d), 3))
Z = StandardScaler().fit_transform(L[cols].values); lam = np.linalg.eigvalsh(Z.T @ Z)
print('5-3d ridge w', np.round(Ridge(alpha=1.0).fit(Z, yy).coef_, 0))
e = sm.datasets.engel.load_pandas().data; lx = np.log(e.income.values); ly = np.log(e.foodexp.values)
m = make_pipeline(StandardScaler(), SGDRegressor(eta0=0.01, max_iter=1000, tol=1e-6, penalty=None, random_state=0)).fit(lx[:, None], ly)
print('5-4d sgd slope in log units', round(m[-1].coef_[0] / lx.std(), 4), 'epochs', m[-1].n_iter_, 'exact', round(we, 4))

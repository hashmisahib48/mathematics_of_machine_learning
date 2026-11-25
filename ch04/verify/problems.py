"""Answers to the problems of Chapter 4 (solution manual)."""
import numpy as np, statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict, KFold, KFold
d = sm.datasets.longley.load_pandas().data; d.to_csv('../data/longley.csv', index=False)
cols = ['GNPDEFL', 'GNP', 'UNEMP', 'ARMED', 'POP', 'YEAR']; X = d[cols].values; y = d.TOTEMP.values
tr = (d.YEAR < 1962).values
def fit(a, rows):
    est = Ridge(alpha=a) if a else LinearRegression()
    return make_pipeline(StandardScaler(), est).fit(X[rows], y[rows])
for a in (0, 0.1, 1, 10):
    f = fit(a, np.ones(len(y), bool)); m = fit(a, tr); p62 = m.predict(X[~tr])[0]
    print('4-1 alpha', a, 'w(all)', f[-1].coef_.round(0), 'w(<1962)', m[-1].coef_.round(0), 'pred1962', round(p62, 0),
          'actual', y[~tr][0], 'pct error', round(100 * (p62 - y[~tr][0]) / y[~tr][0], 2))
print('4-1d diagonal of A+aI (A_jj = n = 16 for StandardScaler):', [(a, 16 + a, round((16 + a) / 16, 4)) for a in (0.1, 1, 10)])
s = sm.datasets.stackloss.load_pandas().data; s.to_csv('../data/stackloss.csv', index=False)
X = s[['AIRFLOW', 'WATERTEMP', 'ACIDCONC']].values; y = s.STACKLOSS.values
for name, m in [('linear', LinearRegression()), ('quad LS', make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), LinearRegression())),
                ('quad ridge 1', make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(1.0))),
                ('quad ridge 10', make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(10.0)))]:
    p = cross_val_predict(m, X, y, cv=LeaveOneOut()); print('4-2', name, 'LOO RMSE', round(np.sqrt(((y - p) ** 2).mean()), 3), 'train R2', round(m.fit(X, y).score(X, y), 4))
e = sm.datasets.engel.load_pandas().data; x = e.income.values[:, None] / 1000; yy = e.foodexp.values
for dgr in (1, 2, 3):
    m = make_pipeline(PolynomialFeatures(dgr, include_bias=False), LinearRegression()); sc = cross_val_score(m, x, yy, cv=KFold(5, shuffle=True, random_state=0), scoring='r2')
    print('4-3 degree', dgr, 'CV R2', round(sc.mean(), 4), 'folds', sc.round(3), 'coef', m.fit(x, yy)[-1].coef_.round(2))
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
for deg in (1, 2, 3):
    m = make_pipeline(PolynomialFeatures(deg, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 29))).fit(a, c)
    print('4-4 degree', deg, 'nfeat', m[0].n_output_features_, 'alpha', round(m[-1].alpha_, 2), 'test R2', round(m.score(b, d), 3))
# folded numerical parts
s = sm.datasets.stackloss.load_pandas().data; x = s.AIRFLOW.values; y = s.STACKLOSS.values
Sxx = ((x - x.mean()) ** 2).sum(); Sxy = ((x - x.mean()) * (y - y.mean())).sum()
for a in (0, 100, 1000):
    w = Sxy / (Sxx + a); b = y.mean() - w * x.mean(); print('4-2d alpha', a, 'w', round(w, 4), 'b', round(b, 4), 'pred(70)', round(w * 70 + b, 3), 'check', Ridge(alpha=a).fit(x[:, None], y).coef_.round(4))
e = sm.datasets.engel.load_pandas().data; xx = e.income.values[:, None] / 1000; yy = e.foodexp.values
m = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(xx, yy); b = m[-1].intercept_; w = m[-1].coef_
print('4-3d b', round(b, 2), 'w', w.round(2), 'pred 1,2,6', [round(b + w[0] * v + w[1] * v * v, 1) for v in (1, 2, 6)], 'peak', round(w[0] / (-2 * w[1]), 3), 'max income', round(xx.max(), 3))
from math import comb
print('4-4 counts C(p+d,d)-1 for p=10:', [comb(10 + k, k) - 1 for k in (1, 2, 3)])
print('4-4 diabetes: n', len(load_diabetes().target), 'features', load_diabetes().feature_names)
print('4-3 Engel: n', len(xx), 'income range', e.income.min(), e.income.max(), 'food range', e.foodexp.min(), e.foodexp.max(), 'households above 3000 francs', int((e.income > 3000).sum()))
for a in (0, 1, 10):
    m = make_pipeline(PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge(alpha=a) if a else LinearRegression()).fit(xx, yy)
    print('4-3e quadratic alpha', a, 'pred at 1,2,6 thousand francs', m.predict(np.array([[1.0], [2.0], [6.0]])).round(1))

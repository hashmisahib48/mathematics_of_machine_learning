import numpy as np, statsmodels.api as sm, math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.metrics import confusion_matrix
# 7-1 spector
d = sm.datasets.spector.load_pandas().data; X = d[['GPA', 'TUCE', 'PSI']].values; y = d.GRADE.values
print('7-1 n', len(d), 'improved', int(y.sum()), 'PSI', int(d.PSI.sum()))
m = LogisticRegression(C=np.inf).fit(X, y); print('7-1 coef', m.coef_.round(4), 'b', m.intercept_.round(4), 'odds', np.exp(m.coef_).round(3))
r = sm.Logit(y, sm.add_constant(X)).fit(disp=0); print('7-1 statsmodels', r.params.round(4), 'p', r.pvalues.round(4))
for psi in (0, 1): print('7-1 P(GPA3,TUCE20,PSI', psi, ')', m.predict_proba([[3.0, 20, psi]])[0, 1].round(4))
print('7-1 LOO acc', cross_val_score(LogisticRegression(C=np.inf), X, y, cv=LeaveOneOut()).mean().round(4))
print('7-1 base rate acc', round(1 - y.mean(), 4))
# 7-2 breast cancer threshold by CV probabilities
X, y = load_breast_cancer(return_X_y=True)
p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=5, method='predict_proba')[:, 0]
for th in (0.5, 0.3, 0.1, 0.05):
    al = p >= th; print('7-2 th', th, 'missed', int(((y == 0) & ~al).sum()), 'false', int(((y == 1) & al).sum()))
# 7-3 digits
D = load_digits(); a, b, c, e = train_test_split(D.data, D.target, test_size=0.25, random_state=0, stratify=D.target)
for C in (0.001, 0.01, 0.1, 1, 10):
    mm = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(a, c); pr = mm.predict(b)
    print('7-3 C', C, 'acc', round((pr == e).mean(), 4), 'errors', int((pr != e).sum()))
mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(a, c); cm = confusion_matrix(e, mm.predict(b))
np.fill_diagonal(cm, 0); i, j = np.unravel_index(cm.argmax(), cm.shape); print('7-3 most common confusion', i, '->', j, cm[i, j])
# numerical parts folded into the case studies
D = sm.datasets.spector.load_pandas().data; b0 = -13.0213; w0 = np.array([2.8261, 0.0952, 2.3787])
for st in (4, 31):
    xr = D[['GPA', 'TUCE', 'PSI']].values[st]; t = b0 + w0 @ xr; print('7-1d student', st + 1, xr, 'score', round(t, 4), 'P', round(1 / (1 + math.exp(-t)), 4), 'grade', D.GRADE[st])
print('7-1e GPA boundary', round((-b0 - 0.0952 * 20 - 2.3787) / 2.8261, 4))
print('7-1f odds factor for +0.5 GPA', round(math.exp(0.5 * 2.8261), 4))
g = D.GPA.values; yv = D.GRADE.values; n = len(g)
gw = ((0.5 - yv) * g).sum() / n; gb = (0.5 - yv).sum() / n
print('7-1g sums: sum x', round(g.sum(), 2), 'sum xy', round((g * yv).sum(), 2), 'sum y', yv.sum(), 'grad w', round(gw, 5), 'grad b', round(gb, 5), 'w1', round(-0.1 * gw, 5), 'b1', round(-0.1 * gb, 5), 'J0', round(math.log(2), 4))
print('7-2d score limits t <= ln((1-p0)/p0)', [(p, round(math.log((1 - p) / p), 4)) for p in (0.5, 0.3, 0.1, 0.05)])
al = p >= 0.1; ms, fl = int(((y == 0) & ~al).sum()), int(((y == 1) & al).sum())
print('7-2d at 0.1 missed', ms, 'of', int((y == 0).sum()), '=', round(ms / (y == 0).sum(), 4), 'flagged', fl, 'of', int((y == 1).sum()), '=', round(fl / (y == 1).sum(), 4))
print('7-3c acc C=0.1', round(1 - 13 / 450, 4), 'params', 10 * 64 + 10)
Dg = load_digits(); a, b, c, e = train_test_split(Dg.data, Dg.target, test_size=0.25, random_state=0, stratify=Dg.target)
mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(a, c); pr = mm.predict(b); bad = np.where(pr != e)[0][0]
T = mm.decision_function(b[[bad]])[0]; Pp = mm.predict_proba(b[[bad]])[0]; o = np.argsort(-T)[:2]
print('7-3d test index', bad, 'true', e[bad], 'pred', pr[bad], 'top classes', o, 'scores', T[o].round(4), 'probs', Pp[o].round(4), 'ratio', round(Pp[o[0]] / Pp[o[1]], 4), 'exp diff', round(math.exp(T[o[0]] - T[o[1]]), 4))
# 7-4 imbalanced iris
from sklearn.datasets import load_iris
Xr, yr = load_iris(return_X_y=True); sel = np.r_[0:100, 140:150]; Xi, yi = Xr[sel][:, 2:], yr[sel]
print('7-4 sizes', np.bincount(yi), 'n', len(yi))
for cw in (None, 'balanced'):
    mm = make_pipeline(StandardScaler(), LogisticRegression(class_weight=cw)); pi = cross_val_predict(mm, Xi, yi, cv=5)
    print('7-4 class_weight', cw, 'confusion\n', confusion_matrix(yi, pi))
print('7-4c weights n/(K n_c)', [round(len(yi) / (3 * k), 4) for k in np.bincount(yi)], 'ratio', round(50 / 10, 4))
pp = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), Xi, yi, cv=5, method='predict_proba')[:, 2]
for th in (0.5, 0.3, 0.2):
    al = pp >= th; print('7-4d th', th, 'virginica found', int((al & (yi == 2)).sum()), 'of 10, flagged versicolor', int((al & (yi == 1)).sum()))
print('7-2e flag threshold for cost ratio 20', round(1 / 21, 4))
print('7-4e term -ln 0.30', round(-math.log(0.30), 4), 'weighted', round(-math.log(0.30) * 110 / 30, 4))
Dg = load_digits(); a, b, c, e = train_test_split(Dg.data, Dg.target, test_size=0.25, random_state=0, stratify=Dg.target)
mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(a, c)
print('7-3e digit 12 P(true class 8)', round(mm.predict_proba(b[[12]])[0, 8], 4), 'cost', round(-math.log(mm.predict_proba(b[[12]])[0, 8]), 4), 'random guess', round(math.log(10), 4))

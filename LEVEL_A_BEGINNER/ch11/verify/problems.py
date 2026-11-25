import sys, math, numpy as np, statsmodels.api as sm
sys.path.insert(0, '../code')
from evaluate import roc_auc
from sklearn.datasets import load_wine, load_diabetes, load_breast_cancer
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_score, GridSearchCV, train_test_split,
                                     StratifiedKFold, LeaveOneOut, cross_val_predict)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, root_mean_squared_error, roc_auc_score, f1_score
# 11-1 wines
X, y = load_wine(return_X_y=True); cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
for n, m in [('knn5', make_pipeline(StandardScaler(), KNeighborsClassifier(5))), ('logistic', make_pipeline(StandardScaler(), LogisticRegression())), ('nb', GaussianNB()), ('tree3', DecisionTreeClassifier(max_depth=3, random_state=0))]:
    s = cross_val_score(m, X, y, cv=cv); print('11-1a', n, round(s.mean(), 4), round(s.std(), 4))
a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(a); print('11-1c proline train mean', round(sc.mean_[12], 2), 'sd', round(sc.scale_[12], 2), 'first test wine proline', b[0, 12], 'z', round((b[0, 12] - sc.mean_[12]) / sc.scale_[12], 4))
sc2 = StandardScaler().fit(X); print('11-1c with all data mean', round(sc2.mean_[12], 2), 'sd', round(sc2.scale_[12], 2), 'z', round((b[0, 12] - sc2.mean_[12]) / sc2.scale_[12], 4))
# 11-2 diabetes
X, y = load_diabetes(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0)
p = Pipeline([('poly', PolynomialFeatures(include_bias=False)), ('sc', StandardScaler()), ('r', Ridge())])
g = GridSearchCV(p, {'poly__degree': [1, 2, 3], 'r__alpha': np.logspace(-2, 3, 11)}, cv=5).fit(a, c)
pr = g.predict(b)
print('11-2 best', g.best_params_, round(g.best_score_, 4), 'test R2', round(g.score(b, d), 4), 'MAE', round(mean_absolute_error(d, pr), 2), 'RMSE', round(root_mean_squared_error(d, pr), 2), 'sd test', round(d.std(), 2))
print('11-2 first three test', d[:3], pr[:3].round(1), 'abs err', np.abs(d[:3] - pr[:3]).round(1))
print('11-2 fits', 3 * 11 * 5 + 1)
# 11-3 nested
X, y = load_breast_cancer(return_X_y=True)
pipe = Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier())]); grid = {'k__n_neighbors': [1, 3, 5, 7, 9, 15, 25], 'k__weights': ['uniform', 'distance']}
inner = StratifiedKFold(5, shuffle=True, random_state=1); outer = StratifiedKFold(5, shuffle=True, random_state=2)
gs = GridSearchCV(pipe, grid, cv=inner).fit(X, y); print('11-3a non-nested best_score', round(gs.best_score_, 4), gs.best_params_)
ns = cross_val_score(GridSearchCV(pipe, grid, cv=inner), X, y, cv=outer); print('11-3b nested', round(ns.mean(), 4), ns.round(4))
sizes = [len(t) for _, t in outer.split(X, y)]
print('11-3d weighted outer mean', round(float((ns * sizes).sum() / sum(sizes)), 4), 'unweighted', round(ns.mean(), 4), 'correct', int(round((ns * sizes).sum())), 'of', sum(sizes))
print('11-3c outer test sizes', [len(t) for _, t in outer.split(X, y)], 'inner train approx', [len(tr) for tr, _ in outer.split(X, y)][0] * 4 // 5, 'fits', 5 * (14 * 5 + 1))
inner2 = StratifiedKFold(5, shuffle=True, random_state=3); outer2 = StratifiedKFold(5, shuffle=True, random_state=4)
gs2 = GridSearchCV(pipe, grid, cv=inner2).fit(X, y); ns2 = cross_val_score(GridSearchCV(pipe, grid, cv=inner2), X, y, cv=outer2)
print('11-3e other divisions: non-nested', round(gs2.best_score_, 4), gs2.best_params_, 'nested', round(ns2.mean(), 4))
# 11-4 spector
D = sm.datasets.spector.load_pandas().data; Xs = D[['GPA', 'TUCE', 'PSI']].values; ys = D.GRADE.values.astype(int)
prob = cross_val_predict(LogisticRegression(C=np.inf, max_iter=10000), Xs, ys, cv=LeaveOneOut(), method='predict_proba')[:, 1]
pred = (prob >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(ys, pred).ravel()
P_ = tp / (tp + fp); R = tp / (tp + fn)
print('11-4 TP FP FN TN', tp, fp, fn, tn, 'acc', round((tp + tn) / 32, 4), 'prec', round(P_, 4), 'rec', round(R, 4), 'F1', round(2 * P_ * R / (P_ + R), 4), 'spec', round(tn / (tn + fp), 4))
print('11-4 AUC own', round(roc_auc(list(ys), list(prob)), 4), 'sk', round(roc_auc_score(ys, prob), 4), 'pairs', int(ys.sum()) * int((1 - ys).sum()))
for th in (0.2, 0.3, 0.4, 0.5, 0.6):
    pp = (prob >= th).astype(int); print('11-4 th', th, 'F1', round(f1_score(ys, pp), 4), 'recall', round(pp[ys == 1].mean(), 4), 'FP', int(((ys == 0) & (pp == 1)).sum()))
print('11-4 always no: acc', round(1 - ys.mean(), 4))
print('11-2d R2 from RMSE (rounded inputs)', round(1 - 56.11 ** 2 / 70.46 ** 2, 4))
s5 = cross_val_score(LogisticRegression(C=np.inf, max_iter=10000), Xs, ys, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print('11-4e 5-fold', s5.round(4), round(s5.mean(), 4), round(s5.std(), 4))

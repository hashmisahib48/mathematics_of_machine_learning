import sys, math, numpy as np, statsmodels.api as sm
sys.path.insert(0, '../code')
from tree import gini, mse, DecisionTreeClassifier as TC, DecisionTreeRegressor as TR
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
# 9-1
W = load_wine(); X, y = W.data, W.target
print('9-1 data', W.data.shape, 'cultivars', np.bincount(W.target))
for d in (1, 2, 3, 4, None): print('9-1 depth', d, round(cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X, y, cv=5).mean(), 4))
m = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y); imp = m.feature_importances_
print('9-1 top', [(W.feature_names[j], round(imp[j], 3)) for j in np.argsort(-imp)[:3]])
# 9-2 engel
e = sm.datasets.engel.load_pandas().data; x = e[['income']].values; f = e.foodexp.values
kf = KFold(5, shuffle=True, random_state=0)
print('9-2 linear CV R2', round(cross_val_score(LinearRegression(), x, f, cv=kf).mean(), 4))
for d in (1, 2, 3, 5, 8): print('9-2 depth', d, 'CV R2', round(cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=0), x, f, cv=kf).mean(), 4))
print('9-2 depth1 split', DecisionTreeRegressor(max_depth=1).fit(x, f).tree_.threshold[0].round(2))
# 9-3 instability
B = load_breast_cancer(); print('9-3 data', B.data.shape, 'classes', np.bincount(B.target)); a, b, c, d = train_test_split(B.data, B.target, test_size=0.25, random_state=0, stratify=B.target)
votes = np.zeros((len(d), 2)); accs = []
for s in range(10):
    aa, cc = resample(a, c, random_state=s); t = DecisionTreeClassifier(random_state=0).fit(aa, cc)
    accs.append(t.score(b, d)); votes[np.arange(len(d)), t.predict(b)] += 1
    print('9-3 sample', s, 'root', B.feature_names[t.tree_.feature[0]], round(t.tree_.threshold[0], 3), 'acc', round(accs[-1], 4))
print('9-3 range', round(min(accs), 4), round(max(accs), 4), 'vote acc', round((votes.argmax(1) == d).mean(), 4))
# numerical parts folded into the case studies
cnt = np.bincount(y_w := W.target); q = cnt / cnt.sum()
print('9-1d counts', cnt, 'gini', round(1 - (q ** 2).sum(), 4), 'entropy', round(-(q * np.log2(q)).sum(), 4))
tw = DecisionTreeClassifier(max_depth=3, random_state=0).fit(W.data, W.target); T = tw.tree_
l, r = T.children_left[0], T.children_right[0]
print('9-1d root', W.feature_names[T.feature[0]], round(T.threshold[0], 3), 'left n', T.n_node_samples[l], 'counts', (T.value[l][0] * T.n_node_samples[l]).round(0), 'gini', round(T.impurity[l], 4), 'right n', T.n_node_samples[r], 'counts', (T.value[r][0] * T.n_node_samples[r]).round(0), 'gini', round(T.impurity[r], 4))
wg = (T.n_node_samples[l] * T.impurity[l] + T.n_node_samples[r] * T.impurity[r]) / T.n_node_samples[0]; print('9-1d weighted', round(wg, 4), 'decrease', round(T.impurity[0] - wg, 4))
t1 = DecisionTreeRegressor(max_depth=1).fit(x, f).tree_
print('9-2c split', round(t1.threshold[0], 2), 'n', t1.n_node_samples[1:], 'means', t1.value.ravel()[1:].round(2), 'root mean', round(t1.value.ravel()[0], 2), 'mse', t1.impurity.round(1))
S = sm.datasets.stackloss.load_pandas().data; xa = S.AIRFLOW.values; ya = S.STACKLOSS.values
v = sorted(set(xa)); print('9-4 root mean', round(ya.mean(), 4), 'mse', round(mse(list(ya)), 4))
for p_, q_ in zip(v, v[1:]):
    s_ = (p_ + q_) / 2; L = ya[xa <= s_]; R = ya[xa > s_]
    print('9-4 split', s_, 'nL', len(L), 'meanL', round(L.mean(), 4), 'nR', len(R), 'meanR', round(R.mean(), 4), 'Isplit', round((len(L) * mse(list(L)) + len(R) * mse(list(R))) / 21, 4))
print('9-4 sums: by group', {g_: (int((xa == g_).sum()), float(ya[xa == g_].sum()), float((ya[xa == g_] ** 2).sum())) for g_ in v})
tr = DecisionTreeRegressor(max_depth=2).fit(xa[:, None], ya); print('9-4c depth2 thresholds', tr.tree_.threshold.round(1), 'values', tr.tree_.value.ravel().round(3), 'pred 60, 72', tr.predict([[60], [72]]).round(3))
print('9-2c weighted mse', round((178 * 19807.1 + 57 * 81424.3) / 235, 1))

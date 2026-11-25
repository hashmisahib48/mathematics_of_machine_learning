import sys, math, numpy as np
sys.path.insert(0, '../code')
from naive_bayes import GaussianNB as GN, MultinomialNB as MN
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score
# 8-1
X, y = load_digits(return_X_y=True); a, b, c, d = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for al in (0.01, 0.1, 1, 10): print('8-1 alpha', al, round(MultinomialNB(alpha=al).fit(a, c).score(b, d), 4))
for t in (0, 4, 8, 12): print('8-1 bern binarize', t, round(BernoulliNB(binarize=t).fit(a, c).score(b, d), 4))
cm = confusion_matrix(d, MultinomialNB().fit(a, c).predict(b)); np.fill_diagonal(cm, 0); i, j = np.unravel_index(cm.argmax(), cm.shape); print('8-1 confusion', i, '->', j, cm[i, j])
# 8-2
B = load_breast_cancer(); X, y = B.data, B.target
print('8-2 corr radius-perimeter', round(np.corrcoef(X[:, 0], X[:, 2])[0, 1], 4), 'radius-area', round(np.corrcoef(X[:, 0], X[:, 3])[0, 1], 4))
for name, cols in [('all 30', list(range(30))), ('10 mean', list(range(10))), ('mean without perimeter, area', [0, 1, 4, 5, 6, 7, 8, 9])]:
    p = cross_val_predict(GaussianNB(), X[:, cols], y, cv=5, method='predict_proba')[:, 1]
    print('8-2', name, 'acc', round(accuracy_score(y, p > 0.5), 4), 'logloss', round(log_loss(y, p), 4))
# 8-3
for n in (10, 20, 40, 80, 160, 320):
    sp = StratifiedShuffleSplit(n_splits=50, train_size=n, test_size=200, random_state=0); r = {'nb': [], 'lr': []}
    for tr, te in sp.split(X, y):
        r['nb'].append(GaussianNB().fit(X[tr], y[tr]).score(X[te], y[te]))
        r['lr'].append(make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr]).score(X[te], y[te]))
    print('8-3 n', n, 'nb', round(np.mean(r['nb']), 4), 'lr', round(np.mean(r['lr']), 4))
# short
# numerical parts folded into the case studies
Xd, yd = load_digits(return_X_y=True); a, b, c, d = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
N0 = a[c == 0].sum(0); print('8-1d class0 total ink', N0.sum(), 'never-inked pixels', int((N0 == 0).sum()), 'theta alpha1', round(1 / (N0.sum() + 64), 8), 'ln', round(math.log(1 / (N0.sum() + 64)), 4))
jl = MultinomialNB().fit(a, c).predict_joint_log_proba(b[[0]])[0]; print('8-1e log joints', jl.round(1), 'max', round(jl.max(), 1), 'log10 e^max', round(jl.max() / math.log(10), 1))
K, p = 10, 64; print('8-1f naive', 2 * K * p + K, 'full', K * p + K * p * (p + 1) // 2 + K)
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
Ii = load_iris(); xl = Ii.data[:, [2]]; ti = Ii.target; gi = GaussianNB().fit(xl, ti)
mu, var = gi.theta_.ravel(), gi.var_.ravel(); print('8-4 means', mu.round(4), 'vars', var.round(4))
L = [math.log(1 / 3) - 0.5 * math.log(2 * math.pi * v) - (4.9 - m) ** 2 / (2 * v) for m, v in zip(mu, var)]
print('8-4a L at 4.9', [round(v, 4) for v in L], 'post', gi.predict_proba([[4.9]]).round(4))
from scipy.optimize import brentq
f = lambda x: (-0.5 * math.log(var[1]) - (x - mu[1]) ** 2 / (2 * var[1])) - (-0.5 * math.log(var[2]) - (x - mu[2]) ** 2 / (2 * var[2]))
print('8-4b boundary ve/vi', round(brentq(f, 4.3, 5.5), 4))
f2 = lambda x: (-0.5 * math.log(var[0]) - (x - mu[0]) ** 2 / (2 * var[0])) - (-0.5 * math.log(var[1]) - (x - mu[1]) ** 2 / (2 * var[1]))
print('8-4b boundary se/ve', round(brentq(f2, 1.5, 4.0), 4))
cvi = StratifiedKFold(5, shuffle=True, random_state=0)
print('8-4d CV petal length', round(cross_val_score(GaussianNB(), xl, ti, cv=cvi).mean(), 4), 'all four', round(cross_val_score(GaussianNB(), Ii.data, ti, cv=cvi).mean(), 4))
print('8-3c priors', round(212 / 569, 4), round(357 / 569, 4)); print('8-4d', gi.predict([[2.0], [2.3], [4.5], [5.0]]))
# added parts
sp = StratifiedShuffleSplit(n_splits=50, train_size=20, test_size=200, random_state=0); r = {'nb': [], 'lr': []}
for tr, te in sp.split(X, y):
    r['nb'].append(GaussianNB().fit(X[tr], y[tr]).score(X[te], y[te]))
    r['lr'].append(make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr]).score(X[te], y[te]))
print('8-3d n 20 std nb', round(np.std(r['nb']), 4), 'lr', round(np.std(r['lr']), 4), 'mean difference', round(np.mean(r['nb']) - np.mean(r['lr']), 4))

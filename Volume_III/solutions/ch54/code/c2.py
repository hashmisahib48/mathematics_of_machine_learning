# Problem 54C.2: filling blank kilometres with the mean of all rows before the split
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import sys
sys.path.insert(0, '/home/claude/mlbook3/ch54/code')
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from cars import make_cars

df = make_cars(2000, 0); km = df.km.values.copy(); r = np.random.RandomState(5); blank = r.rand(2000) < 0.08; km[blank] = np.nan
tr, te = km[:1500], km[1500:]
print("blank cells: %d of 2000 (training %d, test %d)" % (blank.sum(), np.isnan(tr).sum(), np.isnan(te).sum()))
m_all, m_tr, m_te = np.nanmean(km), np.nanmean(tr), np.nanmean(te)
n_tr, n_te = (~np.isnan(tr)).sum(), (~np.isnan(te)).sum()
print("mean of km: all rows %.3f, training rows %.3f, test rows %.3f; standard deviation of km %.2f" % (m_all, m_tr, m_te, np.nanstd(km)))
print("all-rows mean = weighted mean of the two parts: %.3f (weights %d and %d non-blank values, test share %.3f)" % ((n_tr * m_tr + n_te * m_te) / (n_tr + n_te), n_tr, n_te, n_te / (n_tr + n_te)))
print("the filler of the training blanks moves by %.3f thousand km = %.3f sd; each filler contains %.3f of the test mean" % (m_all - m_tr, (m_all - m_tr) / np.nanstd(km), n_te / (n_tr + n_te)))
print("the test blanks are filled with %.3f instead of the training mean %.3f: difference %.3f" % (m_all, m_tr, m_all - m_tr))

# does it matter for a model? price from age and km, ridge, 1500 / 500 split; and 5-fold CV on the 2000 rows
X = np.c_[df.age.values, km]; y = df.price.values
Xf = X.copy(); Xf[:, 1] = np.where(np.isnan(km), m_all, km)                      # filled before the split
leaky = rmse(y[1500:], Ridge().fit(Xf[:1500], y[:1500]).predict(Xf[1500:]))
pipe = make_pipeline(SimpleImputer(), StandardScaler(), Ridge()); honest = rmse(y[1500:], pipe.fit(X[:1500], y[:1500]).predict(X[1500:]))
print("test RMSE of a ridge on age and km: filled first %.4f, imputer in the pipeline %.4f (difference %.4f)" % (leaky, honest, leaky - honest))
kf = KFold(5, shuffle=True, random_state=0); NEG = "neg_root_mean_squared_error"
cl = -cross_val_score(make_pipeline(StandardScaler(), Ridge()), Xf, y, cv=kf, scoring=NEG).mean(); ch = -cross_val_score(pipe, X, y, cv=kf, scoring=NEG).mean()
print("5-fold RMSE: filled first %.4f, in the pipeline %.4f (difference %.4f)" % (cl, ch, cl - ch))

# Section 53.1: thirty measurements, how many do we need?
import numpy as np
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from bcdata import load

Xa, Xb, ya, yb, names = load()
model = make_pipeline(StandardScaler(), LogisticRegression())
print("all 30: test %.4f, 5-fold %.4f" % (
    model.fit(Xa, ya).score(Xb, yb),
    cross_val_score(model, Xa, ya, cv=5).mean()))
for t, Z, lab in ((0.01, Xa, "raw"),
                  (0.01, MinMaxScaler().fit_transform(Xa), "[0,1]")):
    keep = VarianceThreshold(t).fit(Z).get_support().sum()
    print("variance > %.2f on %-5s data keeps %2d of 30" % (t, lab, keep))
F, p = f_classif(Xa, ya)
order = np.argsort(-F)
for j in order[:3]:
    print("%-22s F = %6.1f  p = %.1e" % (names[j], F[j], p[j]))
print("p < 0.05 for %d of 30 features" % (p < 0.05).sum())
R = np.abs(np.corrcoef(Xa, rowvar=False))
iu = np.triu_indices(30, 1)
print("pairs with |r| > 0.9: %d of %d" % ((R[iu] > 0.9).sum(), len(iu[0])))

# Problem 48B.2: scoring the training rows in novelty mode.
import os
import sys
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../../ch48/code")  # sensors.py
from sensors import make_sensors

X, y = make_sensors()                  # 315 readings, 15 faulty
Z = StandardScaler().fit_transform(X)

nov = LocalOutlierFactor(novelty=True).fit(Z)
trn = LocalOutlierFactor(novelty=False)
lab_trn = trn.fit_predict(Z)           # the training-row scores
lab_nov = nov.predict(Z)           # the same rows as "new" ones
a = trn.negative_outlier_factor_   # -LOF, own row excluded
b = nov.score_samples(Z)           # own row is a neighbour at 0
gap = b - a
print("largest |difference|     :", np.abs(gap).max().round(4))
print("rows that differ         :", int((abs(gap) > 1e-9).sum()))
print("rows with b > a          :", int((gap > 1e-9).sum()))
print("median of b - a          :", np.median(gap).round(4))
print("flagged by fit_predict   :", int((lab_trn == -1).sum()))
print("flagged by predict       :", int((lab_nov == -1).sum()))
both = (lab_trn == -1) & (lab_nov == -1)
print("flagged by both          :", int(both.sum()))
print("faults among the flagged :",
      int(y[lab_trn == -1].sum()), int(y[lab_nov == -1].sum()))
worst = np.argmin(a)
print("worst training row: -LOF", a[worst].round(3),
      "as a new row", b[worst].round(3))
print("least odd row: -LOF", a.max().round(3),
      "as a new row", b[np.argmax(a)].round(3))

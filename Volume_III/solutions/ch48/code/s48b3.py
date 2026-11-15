# Problem 48B.3: a wrong contamination (15 faults in 315).
import os
import sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../../ch48/code")  # sensors.py
from sensors import make_sensors

X, y = make_sensors()
Z = StandardScaler().fit_transform(X)
n, nf = len(y), int(y.sum())
print("true share", round(nf / n, 4), " faults", nf, " of", n)
print("gamma  flagged  caught  precision  recall  15/flagged")
for g in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
    f = IsolationForest(contamination=g, random_state=0)
    flag = f.fit_predict(Z) == -1          # -1 = strange
    k = int(flag.sum())
    tp = int((flag & (y == 1)).sum())
    pr, rc = precision_score(y, flag), recall_score(y, flag)
    print(f"{g:5.2f} {k:7d} {tp:7d} {pr:10.4f} {rc:7.4f}"
          f" {nf / k:9.4f}")

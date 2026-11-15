# Problem 48B.1: how many trees make the ranking stable?
import os
import sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../../ch48/code")  # sensors.py
from sensors import make_sensors

X, y = make_sensors()              # 315 readings, 15 faulty
Z = StandardScaler().fit_transform(X)

print("trees  mean rho    sd    t(1/rho-1)  top-15 overlap")
for t in (5, 25, 100, 400):
    rho, same = [], []
    for a in range(8):             # 8 pairs of seeds
        f1 = IsolationForest(n_estimators=t, random_state=2*a)
        f2 = IsolationForest(n_estimators=t, random_state=2*a+1)
        s1 = f1.fit(Z).score_samples(Z)
        s2 = f2.fit(Z).score_samples(Z)
        rho.append(spearmanr(s1, s2)[0])
        top1, top2 = np.argsort(s1)[:15], np.argsort(s2)[:15]
        same.append(len(set(top1) & set(top2)))
    m = np.mean(rho)
    # if rho = 1 / (1 + c/t), then t (1/rho - 1) is a constant c
    c = t * (1 / m - 1)
    print(f"{t:5d}  {m:8.4f} {np.std(rho):7.4f} {c:8.2f}"
          f"  {np.mean(same):10.1f} of 15")

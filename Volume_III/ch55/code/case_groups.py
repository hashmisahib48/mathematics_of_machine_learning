# Section 55.9: patients measured eight times (invented data).
import json, numpy as np
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier
from cvdata import make_patients

rs = dict(shuffle=True, random_state=0)
cvs = {"KFold, shuffled": ms.KFold(5, **rs),
       "StratifiedKFold": ms.StratifiedKFold(5, **rs),
       "GroupKFold": ms.GroupKFold(5),
       "StratifiedGroupKFold": ms.StratifiedGroupKFold(5, **rs),
       "LeaveOneGroupOut": ms.LeaveOneGroupOut()}
rf = RandomForestClassifier(50, random_state=0)
est = {k: [] for k in cvs}
truth = []
for seed in range(8):                 # eight invented hospitals
    X, y, g = make_patients(40, 8, seed=seed)
    Xn, yn, _ = make_patients(150, 8, seed=100 + seed)  # new ones
    for k, cv in cvs.items():
        grp = g if "Group" in k else None   # groups only if used
        est[k].append(ms.cross_val_score(rf, X, y, groups=grp,
                                         cv=cv).mean())
    truth.append(rf.fit(X, y).score(Xn, yn))
t = np.mean(truth)
print(f"{'splitter':22s}{'mean':>7s}{'sd':>7s}{'gap to new':>12s}")
for k, v in est.items():
    m, s = np.mean(v), np.std(v, ddof=1)
    print(f"{k:22s}{m:7.3f}{s:7.3f}{m - t:+12.3f}")
sd = np.std(truth, ddof=1)
print(f"{'new patients (truth)':22s}{t:7.3f}{sd:7.3f}")
# (the figures read these numbers)
out = {k: [float(np.mean(v)), float(np.std(v, ddof=1))]
       for k, v in est.items()}
out["truth"] = [float(t), float(sd)]
json.dump(out, open("verify/case_groups.json", "w"))

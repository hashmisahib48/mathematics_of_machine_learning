# How contamination sets the threshold (Eq. 48.3-1).
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sensors import make_sensors

X, y = make_sensors()
Z = StandardScaler().fit_transform(X)          # 315 readings
print("gamma   IsolationForest    LOF          EllipticEnvelope")
print("        flagged  offset_  flagged offset_  flagged offset_")
for g in ("auto", 0.02, 0.05, 0.10):
    row = f"{g!s:6s}"
    for d in (IsolationForest, LocalOutlierFactor,
              EllipticEnvelope):
        if g == "auto" and d is EllipticEnvelope:
            row += "       -        -"       # float only
            continue
        kw = {} if d is LocalOutlierFactor else {"random_state": 0}
        m = d(contamination=g, **kw)
        lab = m.fit_predict(Z) if hasattr(m, "fit_predict") \
            else m.fit(Z).predict(Z)
        row += f"  {(lab == -1).sum():6d} {m.offset_:8.3f}"
    print(row)
s = IsolationForest(contamination=0.05, random_state=0).fit(Z)
q = np.percentile(s.score_samples(Z), 5)
print("5th percentile of score_samples:", round(q, 4),
      "= offset_:", round(s.offset_, 4))

# The four detectors of the chapter on the pump readings.
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sensors import make_sensors

X, y = make_sensors()                  # 315 readings, 15 faulty
sc = StandardScaler().fit(X)
Z = sc.transform(X)
new = sc.transform([[56, 1.1], [66, 1.8], [70, 4.5]])
dets = {"IsolationForest": IsolationForest(random_state=0),
        "LocalOutlierFactor": LocalOutlierFactor(novelty=True),
        "OneClassSVM": OneClassSVM(nu=0.05),
        "EllipticEnvelope": EllipticEnvelope(contamination=0.05,
                                             random_state=0)}
print(f"{'':20s}{'flagged':>8s}{'offset_':>9s}   new readings")
for name, d in dets.items():
    d.fit(Z)
    n_out = (d.predict(Z) == -1).sum()
    print(f"{name:20s}{n_out:8d}{np.ravel(d.offset_)[0]:9.3f}  ",
          d.decision_function(new).round(3), d.predict(new))
lof = LocalOutlierFactor()         # novelty=False: training only
lab = lof.fit_predict(Z)
print("LOF fit_predict flags", (lab == -1).sum(),
      "; offset_", lof.offset_,
      "; worst -LOF", round(-lof.negative_outlier_factor_.min(), 2))

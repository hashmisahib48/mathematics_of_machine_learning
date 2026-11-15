# Problem 29C.1: why the 5-nearest-neighbour rule has a large log loss
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as sc

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
n = len(yte)
knn = make_pipeline(sc(), KNeighborsClassifier()).fit(Xtr, ytr)
P = knn.predict_proba(Xte); pt = P[np.arange(n), yte]          # probability of the true class
print("values taken by P(true class):", sorted({round(float(v), 1) for v in pt}))
print("errors:", int((knn.predict(Xte) != yte).sum()), " patients with P(true) = 0:", int((pt == 0).sum()),
      " with P(true) = 0.2:", int((np.isclose(pt, 0.2)).sum()), " with P(true) = 0.4:", int(np.isclose(pt, 0.4).sum()))
ll = log_loss(yte, P); print(f"log loss {ll:.4f}")
clip = -np.log(1e-15)                                           # scikit-learn clips probabilities at 1e-15
print(f"cost of one patient with P(true) = 0: -ln(1e-15) = {clip:.2f}; divided by {n}: {clip / n:.4f}; two patients: {2 * clip / n:.4f}")
rest = (ll * n - 2 * clip) / (n - 2)
print(f"mean loss of the other {n - 2} patients: {rest:.4f}   log loss of the 143 if those two had P(true) = 0.2: "
      f"{(ll * n - 2 * clip + 2 * -np.log(0.2)) / n:.4f}")
print("\nremedies (test log loss, test errors):")
def show(name, m):
    m.fit(Xtr, ytr); print(f"  {name:38s} {log_loss(yte, m.predict_proba(Xte)):.4f}  {int((m.predict(Xte) != yte).sum())}")
show("5 neighbours (as in Listing 29.2)", make_pipeline(sc(), KNeighborsClassifier()))
show("15 neighbours", make_pipeline(sc(), KNeighborsClassifier(15)))
show("5 neighbours, weights by distance", make_pipeline(sc(), KNeighborsClassifier(weights="distance")))
show("5 neighbours + sigmoid calibration", make_pipeline(sc(), CalibratedClassifierCV(KNeighborsClassifier(), method="sigmoid", cv=5)))
a = 1.0                                                        # additive smoothing (k + a) / (5 + 2a)
Ps = (P * 5 + a) / (5 + 2 * a); print(f"  {'5 neighbours, counts smoothed (k+1)/7':38s} {log_loss(yte, Ps):.4f}  {int((Ps.argmax(1) != yte).sum())}")

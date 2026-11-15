# Twenty forecasts at three levels: reliability and the Brier score.
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

q_lv = np.array([0.9, 0.5, 0.1])        # forecast levels
n_k = np.array([10, 6, 4])              # cases per level
hits = np.array([7, 3, 1])              # events per level
q = np.repeat(q_lv, n_k)
y = np.concatenate([[1] * h + [0] * (m - h)
                    for h, m in zip(hits, n_k)])

o_k = hits / n_k                        # observed frequencies
o = y.mean()
rel = np.sum(n_k * (q_lv - o_k) ** 2) / len(y)      # Eq. 32.4-3
res = np.sum(n_k * (o_k - o) ** 2) / len(y)
unc = o * (1 - o)
print("level  n  hits  observed")
for a, b, c, d in zip(q_lv, n_k, hits, o_k):
    print(f"{a:5.1f} {b:2d} {c:5d}  {d:8.2f}")
print(f"REL {rel:.4f}  RES {res:.4f}  UNC {unc:.4f}")
print(f"REL - RES + UNC = {rel - res + unc:.4f}")
print(f"mean (y - q)^2   = {np.mean((y - q) ** 2):.4f}")
print(f"brier_score_loss = {brier_score_loss(y, q):.4f}")
t, p = calibration_curve(y, q, n_bins=5)
print("calibration_curve prob_true:", t, " prob_pred:", p)

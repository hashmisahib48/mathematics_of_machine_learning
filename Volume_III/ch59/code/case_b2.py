# Section 59.5 (continued): the test rows once, then importance.
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from case_b1 import Xtr, ytr, test, hgb, log
from house import make_houses

leak = make_houses().drop_duplicates().loc[Xtr.index, ["agent_fee"]]
r2 = cross_val_score(Ridge(), leak, ytr, cv=5).mean()
print(f"agent_fee alone, 5-fold R2: {r2:.3f}")

final = log(hgb()).fit(Xtr, ytr)
Xte, yte = test.drop(columns="price"), test["price"]
pred = final.predict(Xte)
print(f"test R2 {r2_score(yte, pred):.3f}   "
      f"MAE {mean_absolute_error(yte, pred):.1f}")
rel = np.median(abs(pred - yte) / yte)
print(f"mean price {yte.mean():.1f}, mean prediction "
      f"{pred.mean():.1f}, median relative error {rel:.3f}")

imp = permutation_importance(
    final, Xte, yte, n_repeats=10, random_state=0,
    scoring="neg_mean_absolute_error")
print("rise of the MAE when one column is shuffled (10 repeats):")
for j in np.argsort(-imp.importances_mean):
    print(f"  {Xte.columns[j]:10s}{imp.importances_mean[j]:7.1f}"
          f" +- {imp.importances_std[j]:.1f}")

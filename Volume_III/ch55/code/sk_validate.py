# Section 55.7: cross_validate, cross_val_predict, and a trap.
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import (KFold, ShuffleSplit,
    cross_validate, cross_val_predict)

X, y = load_diabetes(return_X_y=True)
cv = KFold(5, shuffle=True, random_state=0)
res = cross_validate(Ridge(alpha=1.0), X, y, cv=cv,
    scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
    return_train_score=True, return_estimator=True,
    return_indices=True)
print(*sorted(res))
print("test r2   ", res["test_r2"].round(3))
print("train r2  ", res["train_r2"].round(3))

pred = cross_val_predict(Ridge(alpha=1.0), X, y, cv=cv)
print("mean of fold R2         ", round(res["test_r2"].mean(), 4))
print("R2 of pooled predictions", round(r2_score(y, pred), 4))

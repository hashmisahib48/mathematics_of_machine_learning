# Diabetes: a transformed target, predictions in original units.
import json
import numpy as np
from scipy.stats import skew
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor as TTR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import root_mean_squared_error, r2_score

X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)
print("skewness of y:", skew(ytr).round(3), " range:", y.min(), y.max())
models = {
    "none": LinearRegression(),
    "log": TTR(LinearRegression(), func=np.log, inverse_func=np.exp),
    "Yeo-Johnson": TTR(LinearRegression(),
                       transformer=PowerTransformer()),
    "Box-Cox": TTR(LinearRegression(),
                   transformer=PowerTransformer(method="box-cox")),
}
res = {}
print(f"{'target':>12}{'RMSE':>8}{'R2':>8}{'bias':>8}")
for k, m in models.items():
    p = m.fit(Xtr, ytr).predict(Xte)
    rm = root_mean_squared_error(yte, p)
    r2, bias = r2_score(yte, p), (p - yte).mean()
    res[k] = [round(rm, 2), round(r2, 4), round(bias, 2)]
    print(f"{k:>12}{rm:8.2f}{r2:8.4f}{bias:8.2f}")
lam = models["Yeo-Johnson"].transformer_.lambdas_
print("lambda (Yeo-Johnson):", lam.round(3))
res["lam_yj"] = float(lam[0])
res["lam_bc"] = float(models["Box-Cox"].transformer_.lambdas_[0])
res["skew"] = float(skew(ytr))
json.dump(res, open("verify/case_diabetes.json", "w"), indent=1)

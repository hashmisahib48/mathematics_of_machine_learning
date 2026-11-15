# Extra numbers for Problems 57B.2 and 57B.3 quoted in the solutions
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

print("== 57B.2: F2 from the pooled counts of the folds")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0); f2 = make_scorer(fbeta_score, beta=2, pos_label=0)
models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression()), "forest": RandomForestClassifier(random_state=0)}
sc = {}
for name, m in models.items():
    pr = cross_val_predict(m, Xtr, ytr, cv=cv); TP = ((pr == 0) & (ytr == 0)).sum(); FN = ((pr == 1) & (ytr == 0)).sum(); FP = ((pr == 0) & (ytr == 1)).sum()
    sc[name] = cross_val_score(m, Xtr, ytr, cv=cv, scoring=f2)
    print(f"{name}: recall {TP / (TP + FN):.4f} precision {TP / (TP + FP):.4f} F2 = 5TP/(5TP+4FN+FP) = {5 * TP}/{5 * TP + 4 * FN + FP} = {5 * TP / (5 * TP + 4 * FN + FP):.4f}")
d = sc["logistic"] - sc["forest"]
print("paired differences: mean %.4f  sd %.4f  standard error %.4f  t = %.2f" % (d.mean(), d.std(ddof=1), d.std(ddof=1) / np.sqrt(5), d.mean() / (d.std(ddof=1) / np.sqrt(5))))

print("\n== 57B.3: the price of the one wrong forecast and of the sure ones")
ytr, yte = 1 - ytr, 1 - yte
p = RandomForestClassifier(random_state=0).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
sure = (p == 0) | (p == 1); wrong = sure & ((p == 1) != (yte == 1)); right_sure = sure & ~wrong
print("sure forecasts", sure.sum(), " wrong", wrong.sum(), " right", right_sure.sum(), " others", (~sure).sum(), " n", len(p))
for eps in (np.finfo(float).eps, 0.005, 0.01, 0.02, 0.05):
    w = -np.log(eps) / len(p); r = right_sure.sum() * -np.log(1 - eps) / len(p)
    oth = log_loss(yte[~sure], np.clip(p[~sure], eps, 1 - eps), labels=[0, 1]) * (~sure).sum() / len(p)
    print(f"eps {eps:9.3g}: wrong forecast {w:.4f}  right sure forecasts {r:.4f}  the other {(~sure).sum()} patients {oth:.4f}  total {w + r + oth:.4f}")
print("smallest nonzero forecast", p[p > 0].min(), " largest below 1", p[p < 1].max())

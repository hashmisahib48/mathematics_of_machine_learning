# Numbers quoted in the prose of Chapter 57 (not printed in the book).
import json
import numpy as np
from scipy.stats import norm
from sklearn import metrics as M
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from screendata import make_screen, auc_binormal

N = {}
# 1. binormal screen: exact values at the threshold 1.5
tpr, fpr = norm.cdf(2 - 1.5), norm.sf(1.5)
N["auc_exact"] = auc_binormal(2.0)
N["tpr15"], N["fpr15"] = tpr, fpr
for pi in (0.05, 0.01):
    ppv = tpr * pi / (tpr * pi + fpr * (1 - pi))
    odds = tpr / fpr * pi / (1 - pi)
    N[f"ppv_{pi}"] = ppv
    N[f"ppv_odds_check_{pi}"] = odds / (1 + odds)
N["lr_plus15"] = tpr / fpr
s, y = make_screen(2000, 0.05, 2.0, 0)
N["auc_sample"] = M.roc_auc_score(y, s)
N["ap_sample"] = M.average_precision_score(y, s)
big_s, big_y = make_screen(400000, 0.05, 2.0, 1)
N["auc_big"] = M.roc_auc_score(big_y, big_s)
# 2. log-loss clipping
N["clip_loss"] = -np.log(np.finfo(np.float64).eps)
# 3. breast cancer test set: bootstrap interval of the AUC, Eq. 57.3 (bootstrap)
X, yy = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.25,
                                      random_state=0, stratify=yy)
ytr, yte = 1 - ytr, 1 - yte
lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr, ytr)
sc = lr.decision_function(Xte)
A = M.roc_auc_score(yte, sc)
n1, n0 = int(yte.sum()), int((1 - yte).sum())
Q1, Q2 = A / (2 - A), 2 * A**2 / (1 + A)
se = np.sqrt((A * (1 - A) + (n1 - 1) * (Q1 - A**2)
              + (n0 - 1) * (Q2 - A**2)) / (n1 * n0))
rng = np.random.RandomState(0)
bs = []
for _ in range(2000):
    i = rng.randint(0, len(yte), len(yte))
    bs.append(M.roc_auc_score(yte[i], sc[i]))
N.update(auc_test=A, n_pos=n1, n_neg=n0, hm_se=se,
         boot_se=float(np.std(bs, ddof=1)),
         boot_lo=float(np.percentile(bs, 2.5)),
         boot_hi=float(np.percentile(bs, 97.5)))
acc = []
for _ in range(2000):
    i = rng.randint(0, len(yte), len(yte))
    acc.append((lr.predict(Xte)[i] == yte[i]).mean())
N["acc_boot_lo"], N["acc_boot_hi"] = [float(v) for v in
                                      np.percentile(acc, [2.5, 97.5])]
# 4. R2 against explained variance with a constant bias
Xd, yd = load_diabetes(return_X_y=True)
Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.25, random_state=0)
p = LinearRegression().fit(Xa, ya).predict(Xb) + 20
N["r2_bias"] = M.r2_score(yb, p)
N["ev_bias"] = M.explained_variance_score(yb, p)
N["bias_term"] = np.mean(yb - p) ** 2 / np.var(yb)
N["var_yte"] = float(np.var(yb))
N["rbar_shifted"] = float(np.mean(yb - p))
N["rbar_squared"] = float(np.mean(yb - p) ** 2)
N["rbar_unshifted"] = float(np.mean(yb - p + 20))
# 5. expected scores of a forecaster who says q when the truth is 0.7
pi = 0.7
for q in (0.7, 0.9, 0.5):
    N[f"brier_q{q}"] = (pi - q) ** 2 + pi * (1 - pi)
    N[f"log_q{q}"] = -(pi * np.log(q) + (1 - pi) * np.log(1 - q))
json.dump(N, open("verify/prose_numbers.json", "w"), indent=1)
for k, v in N.items():
    print(f"{k:22s}{v:.5f}")

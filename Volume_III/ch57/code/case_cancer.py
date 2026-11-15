# Section 57.11: breast-cancer models of Vols I and II, one table.
from sklearn import metrics as M
from sklearn.calibration import CalibratedClassifierCV as Cal
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (GradientBoostingClassifier as GBC,
                              HistGradientBoostingClassifier as HGB,
                              RandomForestClassifier as RF)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler as SS
from sklearn.svm import SVC
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
ytr, yte = 1 - ytr, 1 - yte                  # 1 = malignant
models = {
    "prior": DummyClassifier(strategy="prior"),
    "naive Bayes": GaussianNB(),
    "logistic": make_pipeline(SS(), LogisticRegression()),
    "SVC": make_pipeline(SS(), SVC()),
    "SVC, sigmoid": make_pipeline(SS(), Cal(SVC(), cv=5)),
    "forest": RF(random_state=0),
    "GB": GBC(random_state=0),
    "HGB": HGB(random_state=0),
}
out = {}; print(f"{'':13}{'acc':>7}{'AUC':>7}{'AP':>7}{'MCC':>7}"
                f"{'logloss':>9}{'Brier':>7}{'D2':>8}")
for name, m in models.items():
    yh = m.fit(Xtr, ytr).predict(Xte)
    p = m.predict_proba(Xte)[:, 1] if name != "SVC" else None
    s = m.decision_function(Xte) if p is None else p
    row = [(yh == yte).mean(), M.roc_auc_score(yte, s),
           M.average_precision_score(yte, s),
           M.matthews_corrcoef(yte, yh)]
    if p is not None:
        row += [M.log_loss(yte, p), M.brier_score_loss(yte, p),
                M.d2_log_loss_score(yte, p)]
    cells = [f"{round(v, 4) + 0.0:{w}.4f}" for v, w in zip(
        row, (7, 7, 7, 7, 9, 7, 8))]
    print(f"{name:13s}" + "".join(cells) + "      -      -       -"
          * (p is None))
    out[name] = {"row": row, "score": [float(v) for v in s]}
import json
json.dump({"y": yte.tolist(), "models": out},
          open("verify/case_cancer.json", "w"))

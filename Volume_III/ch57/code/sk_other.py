# Section 57.8: displays and the detection-error-tradeoff curve.
import matplotlib
matplotlib.use("Agg")
from sklearn import metrics as m
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0, stratify=y)
ytr, yte = 1 - ytr, 1 - yte                  # 1 = malignant
lr = make_pipeline(StandardScaler(), LogisticRegression())
lr.fit(Xtr, ytr)
roc = m.RocCurveDisplay.from_estimator(lr, Xte, yte)
pr = m.PrecisionRecallDisplay.from_estimator(lr, Xte, yte)
cm = m.ConfusionMatrixDisplay.from_estimator(lr, Xte, yte,
                                             normalize="true")
print("ROC:", round(roc.roc_auc, 4), len(roc.fpr), "points;  PR:",
      round(pr.average_precision, 4), "prevalence",
      round(pr.prevalence_pos_label, 4))
print("normalised matrix:", cm.confusion_matrix.round(3).tolist())
fpr, fnr, _ = m.det_curve(yte, lr.decision_function(Xte))
print("DET: smallest FPR with FNR <= 0.05:",
      fpr[fnr <= 0.05].min().round(4))

# Fixed and tuned thresholds for the breast-cancer screen.
from bcdata import load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import (FixedThresholdClassifier,
                                     TunedThresholdClassifierCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xa, Xb, ya, yb, names = load()     # 0 = malignant, 1 = benign

def cost(y, pred, c_fn, c_fp, pos_label):   # mean cost per patient
    pos = y == pos_label
    miss = pos & (pred != pos_label)
    alarm = ~pos & (pred == pos_label)
    return (c_fn * miss.sum() + c_fp * alarm.sum()) / len(y)

def spec_at_recall(y, pred, pos_label, target):
    pos = y == pos_label
    if (pred[pos] == pos_label).mean() < target:
        return 0.0                           # recall too low
    return (pred[~pos] != pos_label).mean()  # specificity

model = make_pipeline(StandardScaler(), LogisticRegression())
c10 = make_scorer(cost, greater_is_better=False, c_fn=10,
                  c_fp=1, pos_label=0)
r99 = make_scorer(spec_at_recall, pos_label=0, target=0.99)
tuned = TunedThresholdClassifierCV(model, scoring=c10, cv=5)
rules = {"predict (0.5)": model,
         "fixed 1/11": FixedThresholdClassifier(
             model, threshold=1 / 11, pos_label=0),
         "tuned cost": tuned,
         "tuned recall": TunedThresholdClassifierCV(
             model, scoring=r99, cv=5)}
for name, est in rules.items():
    pred = est.fit(Xa, ya).predict(Xb)
    (hit, miss), (alarm, _) = confusion_matrix(yb, pred)
    print(f"{name:14s} caught {hit}  missed {miss}  alarms "
          f"{alarm:2d}  cost {cost(yb, pred, 10, 1, 0):.4f}")
rec = rules["tuned recall"]
print("best_threshold_", round(tuned.best_threshold_, 4),
      "best_score_", round(tuned.best_score_, 4),
      "recall rule", round(rec.best_threshold_, 4))

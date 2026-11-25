# Moving the threshold for the probability of malignancy.
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
a, b, c, d = train_test_split(X, y, test_size=0.25,
                              random_state=0, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression()).fit(a, c)
pm = m.predict_proba(b)[:, 0]             # P(malignant)
for th in (0.5, 0.3, 0.1, 0.02):
    alarm = pm >= th
    missed = int(((d == 0) & ~alarm).sum())
    false = int(((d == 1) & alarm).sum())
    print(f"threshold {th:4.2f}: missed malignant {missed},"
          f" false alarms {false}")

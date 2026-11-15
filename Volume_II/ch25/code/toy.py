# Three stumps on bootstrap samples of six applicants (Ex. 25.6-1).
import numpy as np
from sklearn.tree import DecisionTreeClassifier

income = np.array([18, 22, 36, 40, 55, 62.])
repaid = np.array([0, 0, 1, 1, 0, 1])
draws = [[0, 2, 4, 4, 5, 5], [1, 1, 3, 3, 3, 5], [0, 0, 0, 1, 2, 3]]
X = income[:, None]

stumps, counts = [], []
for m, idx in enumerate(draws):
    w = np.bincount(idx, minlength=6)     # draws per applicant
    t = DecisionTreeClassifier(max_depth=1, random_state=0)
    t.fit(X, repaid, sample_weight=w)
    stumps.append(t)
    counts.append(w)
    lo = t.predict_proba([[18.0]])[0, 1]
    hi = t.predict_proba([[62.0]])[0, 1]
    print(f"stump {m + 1}: counts {w}, P(repaid) {lo:.2f} if "
          f"x <= {t.tree_.threshold[0]:.1f} else {hi:.2f}")

x_new = np.array([[45.0]])                # a new applicant
p = [t.predict_proba(x_new)[0, 1] for t in stumps]
print("P(repaid) at 45:", np.round(p, 2),
      "-> committee", round(np.mean(p), 4))

print("applicant  y  stumps without it   P     OOB class")
for i in range(6):
    out = [m for m in range(3) if counts[m][i] == 0]
    q = np.mean([stumps[m].predict_proba(X[i:i + 1])[0, 1]
                 for m in out])
    print(f"{i + 1:9d}  {repaid[i]}  {[m + 1 for m in out]!s:18s}"
          f" {q:.2f}   {int(q > 0.5)}")

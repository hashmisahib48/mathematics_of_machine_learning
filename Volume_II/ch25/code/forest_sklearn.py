# A random forest on the ten loan applicants of Chapter 24.
import numpy as np
from sklearn.ensemble import RandomForestClassifier

age = np.array([35, 27, 22, 48, 33, 24, 52, 29, 41, 58])
income = np.array([22, 40, 55, 28, 62, 18, 36, 26, 48, 75])
y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
X = np.c_[age, income]

forest = RandomForestClassifier(
    n_estimators=200, oob_score=True, random_state=0).fit(X, y)
print("trees:", len(forest.estimators_),
      " classes:", forest.classes_)
print("features tried at each split:",
      forest.estimators_[0].max_features_)
print("leaves of the first five trees:",
      [int(t.get_n_leaves()) for t in forest.estimators_[:5]])

new = np.array([[30, 45], [23, 60]])      # applicants A and B
print("P(repaid) for A and B:",
      forest.predict_proba(new)[:, 1].round(3))
print("predicted classes:    ", forest.predict(new))

print("OOB accuracy:", forest.oob_score_)
print("OOB P(repaid), applicants 1-10:")
print(forest.oob_decision_function_[:, 1].round(2))
print("importances (age, income):",
      forest.feature_importances_.round(3))
drawn = forest.estimators_samples_[0]     # indices drawn, tree 0
print("tree 0 drew", len(drawn), "indices,",
      len(np.unique(drawn)), "distinct")

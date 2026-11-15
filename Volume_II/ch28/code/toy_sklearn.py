# The twelve visits of toy.py in scikit-learn: two rounds of stumps.
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

x = np.array([8, 15, 22, 31, 40, 47, 58, 66, 73, 85, 92, 110.])
y = np.array([0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1])
model = HistGradientBoostingClassifier(
    max_iter=2, max_leaf_nodes=2, min_samples_leaf=1, max_bins=4,
    learning_rate=0.5, early_stopping=False)
model.fit(x.reshape(-1, 1), y)

print("thresholds", model._bin_mapper.bin_thresholds_[0])
print("baseline raw score %.4f" % model._baseline_prediction[0, 0])
for m, (tree,) in enumerate(model._predictors, 1):
    s, l, r = tree.nodes[:3]
    print("tree %d: x <= %.1f  gain %.4f  leaves %.4f %.4f" % (
        m, s["num_threshold"], s["gain"], l["value"], r["value"]))

raw = model.decision_function(x.reshape(-1, 1))
print("raw scores", raw.round(3))
staged = model.staged_predict_proba(x.reshape(-1, 1))
for m, p in enumerate(staged, 1):
    print("p after tree", m, p[:, 1].round(4))
print("n_iter_", model.n_iter_, " trees per iteration",
      model.n_trees_per_iteration_)

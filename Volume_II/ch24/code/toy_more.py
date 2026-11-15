# A regression split by hand; weakest-link pruning of the loan tree.
import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import DecisionTreeRegressor as DTR

inc = np.array([20, 30, 40, 50, 60, 70.])    # income, thousands
lim = np.array([4, 5, 6, 12, 13, 14.])       # credit limit granted
sse = lambda v: np.sum((v - v.mean()) ** 2)  # Eq. 24.4-2
print("SSE of the parent:", sse(lim))
for t in (inc[:-1] + inc[1:]) / 2:
    L, R = lim[inc <= t], lim[inc > t]
    print(f"  <= {t:4.1f}  means {L.mean():5.2f} {R.mean():5.2f}"
          f"  SSE {sse(L) + sse(R):6.2f}")
reg = DTR(max_depth=1).fit(inc[:, None], lim)
print("sklearn:", reg.tree_.threshold[0],
      reg.tree_.value.ravel()[1:])

X = np.array([[35, 22], [27, 40], [22, 55], [48, 28], [33, 62],
              [24, 18], [52, 36], [29, 26], [41, 48], [58, 75]])
y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 1, 1])
t = DTC(random_state=0).fit(X, y).tree_
r = t.weighted_n_node_samples * t.impurity / len(y)   # R(t)
print("R(t) of the nodes:", r.round(4))
a_root = (r[0] - r[[1, 3, 4]].sum()) / (3 - 1)        # Eq. 24.5-3
a_two = (r[2] - r[[3, 4]].sum()) / (2 - 1)
print("alpha: root", round(a_root, 4), " node 2", round(a_two, 4))
after = (r[0] - r[[1, 2]].sum()) / (2 - 1)    # node 2 pruned away
print("root after node 2 is pruned:", round(after, 4))
path = DTC(random_state=0).cost_complexity_pruning_path(X, y)
print("sklearn alphas:", path.ccp_alphas.round(4))
print("sklearn impurities:", path.impurities.round(4))

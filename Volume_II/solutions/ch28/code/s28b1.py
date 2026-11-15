# Problem 28B.1: HistStump (Listing 28.2) against scikit-learn
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here + "/../../../ch28/code")
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from hist_scratch import HistStump           # class of Listing 28.2

X, y = load_breast_cancer(return_X_y=True)
p = y.mean()                                 # starting probability
g, h = p - y, np.full(len(y), p * (1 - p))   # gradients, hessians

gain = {}
print("        mine: feat  thr     gain   left "
      "| sklearn: feat thr  gain")
for B in (4, 8, 16, 64, 255):
    me = HistStump(max_bins=B, l2=5.0).fit(X, g, h)
    t = np.append(me.thr_[me.j_], np.inf)[me.k_]
    sk = HGBC(max_iter=1, max_leaf_nodes=2, max_bins=B,
              l2_regularization=5.0, learning_rate=1.0,
              early_stopping=False).fit(X, y)
    n = sk._predictors[0][0].nodes[0]
    gain[B] = me.gain_
    print(f"{B:4d}  {me.j_:11d} {t:6.1f} {me.gain_:9.4f} "
          f"{int(me.sums_[0][2]):5d} | {n['feature_idx']:11d} "
          f"{n['num_threshold']:5.1f} {n['gain']:9.4f}")
print("\ngain at 255 bins / gain at 16 bins:",
      round(gain[255] / gain[16], 4))

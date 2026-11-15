# Edges of the plateau of gamma on the rings (Problem 39B.1): bisection on a log scale, success = ARI > 0.99
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../ch39/code'))
warnings.simplefilter('ignore', UserWarning); warnings.simplefilter('ignore', RuntimeWarning)
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
from data import rings

X, y = rings()
good = lambda g: ari(y, SpectralClustering(2, gamma=g, random_state=0).fit_predict(X)) > 0.99
lo, hi = 10.0, 14.0                       # fails at 10, succeeds at 14
for _ in range(8):
    mid = np.sqrt(lo * hi); lo, hi = (lo, mid) if good(mid) else (mid, hi)
print("lower edge of the plateau: gamma = %.2f (fails at %.2f)" % (hi, lo))
lo, hi = 800.0, 1000.0                    # succeeds at 800, fails at 1000
for _ in range(8):
    mid = np.sqrt(lo * hi); lo, hi = (mid, hi) if good(mid) else (lo, mid)
print("upper edge of the plateau: gamma = %.0f (fails at %.0f)" % (lo, hi))
print("check: gamma = 18 and 560 (the values quoted in Section 39.10):", good(18.0), good(560.0))

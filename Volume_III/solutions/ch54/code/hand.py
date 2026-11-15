# Hand checks for Problems 54A.1 and 54A.2 (Chapter 54 solutions)
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
import sys
sys.path.insert(0, '/home/claude/mlbook3/ch54/code')
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cars import make_cars

# 54A.1  median and mean after a log target
mu, s2 = 2.0, 0.25
med = np.exp(mu); fac = np.exp(s2 / 2); mean = med * fac
print("54A.1 median e^2 = %.4f; exp(sigma^2/2) = e^%.3f = %.4f; mean = %.4f (= e^%.3f = %.4f)" % (med, s2 / 2, fac, mean, mu + s2 / 2, np.exp(mu + s2 / 2)))
print("      difference %.4f = %.2f per cent of the median; the median is %.2f per cent below the mean" % (mean - med, 100 * (mean / med - 1), 100 * (1 - med / mean)))
print("      wrong use of sigma = 0.25 instead of sigma^2: factor %.4f" % np.exp(0.25 ** 2 / 2))

# 54A.2  output columns of a ColumnTransformer
X = make_cars(2000, 0).drop(columns="price")
print("\n54A.2 columns:", X.columns.tolist(), " values of colour %d, fuel %d" % (X.colour.nunique(), X.fuel.nunique()))
for lab, rem, drop in (("(a) drop", "drop", None), ("(b) passthrough", "passthrough", None), ("(c) passthrough, drop first", "passthrough", "first")):
    ct = ColumnTransformer([("c", OneHotEncoder(drop=drop), ["colour", "fuel"]), ("n", StandardScaler(), ["age"])], remainder=rem).fit(X)
    out = ct.transform(X); print("      %-28s width %2d  dtype %-8s names %s" % (lab, out.shape[1], out.dtype, ct.get_feature_names_out().tolist()[-4:]))
nc, nf = X.colour.nunique(), X.fuel.nunique()
print("      by hand: (a) %d + %d + 1 = %d; (b) %d + 3 = %d; (c) (%d - 1) + (%d - 1) + 1 + 3 = %d" % (nc, nf, nc + nf + 1, nc + nf + 1, nc + nf + 4, nc, nf, nc + nf + 2))

# a ridge on the output of (b) fails because trim and city pass through as strings
from sklearn.linear_model import Ridge
df = make_cars(2000, 0); ct = ColumnTransformer([("c", OneHotEncoder(), ["colour", "fuel"]), ("n", StandardScaler(), ["age"])], remainder="passthrough").fit(X)
try: Ridge().fit(ct.transform(X), df.price)
except ValueError as e: print("      Ridge on the output of (b): ValueError:", e)

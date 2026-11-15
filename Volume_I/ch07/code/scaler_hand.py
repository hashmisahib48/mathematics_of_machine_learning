# StandardScaler on four numbers, and why the test set is special.
import numpy as np
from sklearn.preprocessing import StandardScaler

x = np.array([[2.0], [4.0], [6.0], [8.0]])     # one feature, n = 4
mu, sigma = x.mean(), x.std()                  # Eq. 7.6-1, ddof=0
print("mean, std (ddof=0) :", mu, round(float(sigma), 4))
print("std with n - 1     :", round(float(x.std(ddof=1)), 4))
z = (x - mu) / sigma                           # Eq. 7.6-2
print("z by hand          :", np.round(z.ravel(), 4))
sc = StandardScaler().fit(x)
print("mean_, scale_      :", sc.mean_, np.round(sc.scale_, 4))
print("transform equals   :", np.allclose(sc.transform(x), z))
print("inverse_transform  :", sc.inverse_transform(z).ravel())
new = np.array([[10.0]])                       # a later sample
print("\nvalue 10, stored mean and std :",
      np.round(sc.transform(new).ravel(), 4))
leaky = StandardScaler().fit(np.vstack([x, new]))
print("value 10, scaler saw it too   :",
      np.round(leaky.transform(new).ravel(), 4))
const = StandardScaler().fit(np.full((3, 1), 5.0))   # no spread
print("constant column: var_, scale_ =", const.var_, const.scale_)
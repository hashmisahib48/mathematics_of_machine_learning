# IsotonicRegression on the dose data of Section 32.1.
import numpy as np
from sklearn.isotonic import IsotonicRegression, isotonic_regression

dose = np.arange(1, 9.0)
y = np.array([2, 5, 3, 4, 1, 6, 5, 7.0])
m = IsotonicRegression().fit(dose, y)
print("X_thresholds_:", m.X_thresholds_)
print("y_thresholds_:", m.y_thresholds_)
print("X_min_, X_max_:", m.X_min_, m.X_max_, m.increasing_)
q = [0.5, 1, 2.5, 5.5, 8, 9]
print("predict at", q)
print("  out_of_bounds='nan'  :", m.predict(q))
clip = IsotonicRegression(out_of_bounds="clip").fit(dose, y)
print("  out_of_bounds='clip' :", clip.predict(q))
try:
    r = IsotonicRegression(out_of_bounds="raise").fit(dose, y)
    r.predict([9])
except ValueError as e:
    print("  out_of_bounds='raise':", str(e)[:48])
b = IsotonicRegression(y_min=3, y_max=6).fit(dose, y)
print("y_min=3, y_max=6 :", b.predict(dose))
w = [1, 1, 1, 1, 4, 1, 1, 1]
print("weight 4 on dose 5:",
      isotonic_regression(y, sample_weight=w))
dec = IsotonicRegression(increasing="auto").fit(dose, -y - dose)
print("auto, decreasing data: increasing_ =", dec.increasing_)
print("transform = predict:", np.allclose(m.transform(q[1:5]),
                                          m.predict(q[1:5])))

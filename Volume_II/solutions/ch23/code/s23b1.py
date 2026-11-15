# Problem 23B.1: the delivery times in seconds.
import numpy as np
from sklearn.svm import SVR

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([3.0, 4.0, 6.0, 7.0, 10.0])      # minutes
s = 60.0                               # seconds per minute


def fit(target, C, eps):
    m = SVR(kernel="linear", C=C, epsilon=eps, tol=1e-10)
    return m.fit(X, target)


def show(name, m):
    w, b = m.coef_[0, 0], m.intercept_[0]
    f6 = m.predict([[6.0]])[0]
    print(f"{name:18s} w={w:7.3f} b={b:8.3f} f(6)={f6:8.3f}")


m_min = fit(y, 1, 1)               # minutes, C = 1, eps = 1
m_def = fit(s * y, 1, 1)           # seconds, same settings
m_sca = fit(s * y, s, s)           # seconds, C = 60, eps = 60
m_eq = fit(y, 1 / s, 1 / s)        # minutes, C/60 and eps/60
show("minutes (1, 1)", m_min)
show("seconds (1, 1)", m_def)
show("seconds (60, 60)", m_sca)
d = np.abs(m_sca.predict(X) - s * m_min.predict(X)).max()
print("max |sec(60,60) - 60 x min(1,1)|  =", f"{d:.1e}")
d = np.abs(m_def.predict(X) - s * m_eq.predict(X)).max()
print("max |sec(1,1) - 60 x min(1/60,1/60)| =", f"{d:.1e}")

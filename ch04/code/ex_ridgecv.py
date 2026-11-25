# Choosing alpha by cross-validation with RidgeCV.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
alphas = np.logspace(-2, 4, 25)              # 0.01 to 10,000
m = make_pipeline(PolynomialFeatures(2, include_bias=False),
                  StandardScaler(), RidgeCV(alphas=alphas))
m.fit(X_tr, y_tr)
cv = m[-1]
print("chosen alpha:", round(cv.alpha_, 2))
print("largest |w|:", round(np.abs(cv.coef_).max(), 2))
print("test R2:", round(m.score(X_te, y_te), 3))

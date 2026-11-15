# The eight-point toy set with scikit-learn (Listing 18.4).
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[1, 1], [3, 3], [1, 2], [3, 2],      # class 0
              [4, 2], [6, 4], [5, 2], [5, 4]], float)   # class 1
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

lda = LinearDiscriminantAnalysis(store_covariance=True)
lda.fit(X, y)
print("priors_      =", lda.priors_)
print("means_       =", lda.means_.tolist())
print("covariance_  =", lda.covariance_.tolist())
print("coef_        =", np.round(lda.coef_, 4))
print("intercept_   =", np.round(lda.intercept_, 4))
print("scalings_    =", np.round(lda.scalings_[:, 0], 4))
print("ratio        =", lda.explained_variance_ratio_)
print("transform    =", np.round(lda.transform(X)[:, 0], 3))
new = np.array([[4.0, 3.0], [3.0, 2.0]])
print("predict      =", lda.predict(new))
print("predict_proba=", np.round(lda.predict_proba(new)[:, 1], 4))
print("decision     =", np.round(lda.decision_function(new), 4))

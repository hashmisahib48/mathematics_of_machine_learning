# Hyperparameters go in; learned attributes come out.
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True, as_frame=True)  # pandas
model = LogisticRegression(C=10.0, max_iter=500)
print(model)                       # only changed values shown
try:
    model.predict(X)
except NotFittedError:
    print("predict before fit: NotFittedError")
same = model.fit(X, y) is model    # learn; fit returns the model
print("fit returns the model itself:", same)
print("coef_ shape   :", model.coef_.shape)       # (K, p)
print("intercept_    :", model.intercept_.shape)   # (K,)
print("classes_      :", model.classes_)
print("n_features_in_:", model.n_features_in_)
print("feature_names_in_:", model.feature_names_in_[:2], "...")
bad = LogisticRegression(C=-1.0)   # no error yet
try:
    bad.fit(X, y)
except ValueError as err:
    print("fit rejects C=-1:", type(err).__name__)
twin = clone(model)                # same settings, not fitted
print("clone: C =", twin.C, " fitted:", hasattr(twin, "coef_"))
model.set_params(C=0.1)            # change a hyperparameter
print("set_params(C=0.1): coef_ still there:",
      hasattr(model, "coef_"))
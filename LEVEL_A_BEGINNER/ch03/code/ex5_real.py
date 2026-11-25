# The diabetes data bundled with scikit-learn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split as split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = split(X, y, test_size=0.25, random_state=0)

model = LinearRegression().fit(X_tr, y_tr)
y_pred = model.predict(X_te)

print("samples, features:", X.shape)
pairs = [f"{n:3s} {c:8.2f}" for n, c in zip(X.columns, model.coef_)]
for k in range(0, 10, 5):
    print("  ".join(pairs[k:k + 5]))
print(f"intercept_ : {model.intercept_:.2f}")
print(f"test R^2   : {r2_score(y_te, y_pred):.3f}")
print(f"test RMSE  : {root_mean_squared_error(y_te, y_pred):.2f}")

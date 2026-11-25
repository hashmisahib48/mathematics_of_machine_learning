# A first model: predict diabetes progression with LinearRegression.
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)      # 442 patients, 10 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          random_state=0)
model = LinearRegression()                     # 1. create
model.fit(X_tr, y_tr)                          # 2. learn from data
y_hat = model.predict(X_te)                    # 3. predict new cases
print("training cases:", len(y_tr), " test cases:", len(y_te))
print("first three predictions:", y_hat[:3].round(1))
print("first three targets:    ", y_te[:3])
r2 = model.score(X_te, y_te)                   # 4. evaluate
print("test R2:", round(r2, 3))

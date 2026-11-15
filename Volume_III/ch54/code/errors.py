# Error messages and silent surprises quoted in Chapter 54.
import numpy as np
from textwrap import fill
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from cars import make_cars

pipe = make_pipeline(StandardScaler(), LogisticRegression())
X3, y3 = np.eye(3), [0, 1, 0]
calls = {
    "short name": lambda: pipe.set_params(C=0.1),
    "wrong order": lambda: make_pipeline(
        LogisticRegression(), StandardScaler()).fit(X3, y3),
    "not fitted": lambda: pipe.predict(X3),
    "transform_input": lambda: make_pipeline(
        StandardScaler(), transform_input=["X_val"]).fit(X3, y3)}
for name, call in calls.items():
    try:
        call()
    except Exception as e:
        head = f"{name}: {type(e).__name__}: "
        print(fill(head + str(e).replace("\n", " "), 66,
                   subsequent_indent="    ", max_lines=2))

X = make_cars(500, 0).drop(columns="price")
ct = ColumnTransformer([("num", StandardScaler(), ["age"]),
                        ("cat", OneHotEncoder(), ["fuel"])])
print("columns in", X.shape[1], "| out",
      ct.fit_transform(X).shape[1], "| remainder:",
      ct.transformers_[-1][1])

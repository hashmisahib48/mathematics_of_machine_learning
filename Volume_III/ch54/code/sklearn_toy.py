from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from cars import make_cars

df = make_cars(2000, 0)
X, y = df.drop(columns="price"), df.price
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     make_column_selector(dtype_exclude="number")),
    ("num", StandardScaler(), ["age", "km"])])
pipe = Pipeline([("prep", prep), ("ridge", Ridge(alpha=1.0))])
pipe.fit(Xtr, ytr)

Z = pipe[:-1].transform(Xte)               # every step but the last
print("output   ", type(Z).__name__, Z.shape, prep.sparse_output_)
print("names    ", prep.get_feature_names_out()[[0, -3, -2, -1]])
print("same object:", pipe.named_steps["prep"] is prep)
test = lambda m: round(rmse(yte, m.fit(Xtr, ytr).predict(Xte)), 3)
print("RMSE, alpha 1 and 100:", test(pipe),
      test(pipe.set_params(ridge__alpha=100.0)))  # step__argument

frozen = FrozenEstimator(StandardScaler().fit(Xtr[["age", "km"]]))
again = clone(make_pipeline(frozen, Ridge()))
print("frozen step kept by clone:", again[0] is frozen)
print("transformers_ in the pipeline, in its clone:",
      hasattr(pipe[0], "transformers_"),
      hasattr(clone(pipe)[0], "transformers_"))

# A pipeline with different preprocessing for different columns.
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

D = load_diabetes(as_frame=True, scaled=False)    # raw units
X, y = D.data, D.target
num = [c for c in X.columns if c != "sex"]
prep = ColumnTransformer([
    ("cat", OneHotEncoder(), ["sex"]),            # sex coded 1 or 2
    ("num", StandardScaler(), num)])
model = Pipeline([("prep", prep), ("ridge", Ridge(alpha=1.0))])
r2 = cross_val_score(model, X, y, cv=5, scoring="r2")
print("cross-validated R2", r2.round(3), " mean", round(r2.mean(), 3))
print("columns after preprocessing:",
      model.fit(X, y)["prep"].get_feature_names_out()[:4])

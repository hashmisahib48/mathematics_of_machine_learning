# A first Pipeline: scaler and classifier behave as one estimator.
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True, as_frame=True)   # DataFrame
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
print("steps          :", [name for name, _ in pipe.steps])
pipe.fit(Xtr, ytr)             # scaler fitted on Xtr only
print("test accuracy  :", round(pipe.score(Xte, yte), 4))
print("scaler mean_   :", pipe[0].mean_.round(3))
keys = [k for k in pipe.get_params() if "__" in k]
print("parameters     :", len(keys), "e.g.", keys[0])
pipe.set_params(logisticregression__C=0.01)      # step__name
pipe.fit(Xtr, ytr)
print("with C = 0.01  :", round(pipe.score(Xte, yte), 4))
scaler = StandardScaler().set_output(transform="pandas")
Z = scaler.fit(Xtr).transform(Xte)
print(type(Z).__name__, Z.shape)
print(Z.iloc[:2, :2].round(2).to_string())
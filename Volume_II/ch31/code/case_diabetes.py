# Case study: disease progression with MLPRegressor.
import warnings
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")           # convergence notes
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                      random_state=0)

def report(name, model):
    model.fit(Xtr, ytr)
    rmse = root_mean_squared_error(yte, model.predict(Xte))
    print(f"{name:26s} train {model.score(Xtr, ytr):7.3f}"
          f"  test {model.score(Xte, yte):7.3f}  RMSE {rmse:5.1f}")

report("least squares", LinearRegression())
report("MLP default, raw y", MLPRegressor(random_state=0))
# standardise the features and the target; undo it for predictions
net = MLPRegressor(hidden_layer_sizes=(16,), solver="lbfgs",
                   alpha=0.1, max_iter=500, random_state=0)
wrap = TransformedTargetRegressor(net, transformer=StandardScaler())
pipe = make_pipeline(StandardScaler(), wrap)
report("MLP (16,), alpha 0.1", pipe)

pre = "transformedtargetregressor__regressor__"
grid = {pre + "hidden_layer_sizes": [(4,), (8,), (16,)],
        pre + "alpha": [0.1, 1, 3, 10]}
search = GridSearchCV(pipe, grid, cv=5).fit(Xtr, ytr)
best = {k.split("__")[-1]: v for k, v in
        search.best_params_.items()}
print("grid search:", best)
print(f"cv R2 {search.best_score_:.3f}  "
      f"test R2 {search.score(Xte, yte):.3f}")

# 8.3  Regression scores by hand and in sklearn; R2 on new data.
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score, root_mean_squared_error

y = np.array([3., 5, 8, 10, 14])       # measured
yh = np.array([4., 5, 7, 12, 12])      # predicted
r = y - yh                             # residuals, Eq. 8.3-1
sse = np.sum(r**2)
sst = np.sum((y - y.mean())**2)
print("residuals", r, " SSE", sse, " SST", sst)
print("MSE ", sse / 5, "  sklearn", mean_squared_error(y, yh))
print("RMSE", round(np.sqrt(sse / 5), 4),
      "  sklearn", round(root_mean_squared_error(y, yh), 4))
print("MAE ", np.mean(np.abs(r)), "  sklearn",
      mean_absolute_error(y, yh))
print("R2  ", round(1 - sse / sst, 4), "  sklearn",
      round(r2_score(y, yh), 4))

x = np.array([1., 2, 3, 4, 5]).reshape(-1, 1)    # delivery data
y = np.array([3., 4, 6, 7, 10])
m = LinearRegression().fit(x, y)
print("training R2:", round(m.score(x, y), 4))
x_new = np.array([6., 7, 8, 9]).reshape(-1, 1)   # longer rides
y_new = np.array([11., 13, 12, 14])
p = m.predict(x_new)
sse = np.sum((y_new - p)**2)
sst = np.sum((y_new - y_new.mean())**2)
print("predictions:", np.round(p, 1))
print("SSE", round(sse, 2), " SST", sst)
print("test R2:", round(r2_score(y_new, p), 3), " = 1 - SSE/SST =",
      round(1 - sse / sst, 3))

# Problem 8C.1: the delivery line on four longer rides
import math
import numpy as np
np.set_printoptions(precision=4, suppress=True)

x = np.array([1., 2, 3, 4, 5]); y = np.array([3., 4, 6, 7, 10])
w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum(); b = y.mean() - w * x.mean()
sse = ((y - (b + w * x)) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
print("training: w = %.3f, b = %.3f, SSE = %.2f, SST = %.1f, R2 = %.4f" % (w, b, sse, sst, 1 - sse / sst))

xn = np.array([6., 7, 8, 9]); yn = np.array([11., 13, 12, 14]); pn = b + w * xn; r = yn - pn
sse_n = (r ** 2).sum(); sst_n = ((yn - yn.mean()) ** 2).sum()
print("new rides: predictions", pn, " residuals", r)
print("SSE = %.2f, mean of new targets = %.1f, SST = %.1f, R2 = %.3f, RMSE = %.3f, MAE = %.3f" % (sse_n, yn.mean(), sst_n, 1 - sse_n / sst_n, math.sqrt(sse_n / 4), np.abs(r).mean()))
print("standard deviation of the new targets (n): %.3f  of the training targets: %.3f" % (yn.std(), y.std()))
wn = ((xn - xn.mean()) * (yn - yn.mean())).sum() / ((xn - xn.mean()) ** 2).sum()
print("slope of the best line through the new rides: %.2f (training line %.2f), intercept %.2f" % (wn, w, yn.mean() - wn * xn.mean()))
sse_b = ((yn - y.mean()) ** 2).sum()
print("constant = training mean %.1f: SSE = %.1f, RMSE = %.2f, R2 against the new mean = %.2f" % (y.mean(), sse_b, math.sqrt(sse_b / 4), 1 - sse_b / sst_n))
print("line against the training mean as baseline: 1 - SSE/SSE_base = %.3f" % (1 - sse_n / sse_b))
print("constant at the new mean 12.5 (needs the new targets): RMSE = %.3f" % math.sqrt(sst_n / 4))

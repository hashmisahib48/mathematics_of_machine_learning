# MyPLS against PLSRegression: coefficients, scores, predictions.
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud
from pls_scratch import MyPLS
from spectra import make_spectra

lin = load_linnerud()
data = {"spectra, 1 target": (*make_spectra(60, 0), (1, 3, 6)),
        "Linnerud, 3 targets": (lin.data, lin.target, (1, 2, 3))}
print(f"{'data':22s}{'A':>3s}{'coef_':>10s}{'scores':>10s}"
      f"{'predict':>10s}")
for name, (X, Y, grid) in data.items():
    for A in grid:
        mine = MyPLS(A).fit(X, Y)
        ref = PLSRegression(A).fit(X, Y)
        d1 = abs(mine.coef_ - ref.coef_).max()
        scores = (X - mine.xm_) / mine.xs_ @ mine.x_rotations_
        d2 = abs(scores - ref.x_scores_).max()
        d3 = abs(mine.predict(X) - ref.predict(X)).max()
        assert max(d1, d2, d3) < 1e-9
        print(f"{name:22s}{A:3d}{d1:10.1e}{d2:10.1e}{d3:10.1e}")

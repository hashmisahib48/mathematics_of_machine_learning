# What the 1.9 series says about the old MDS arguments.
import textwrap
import warnings
import numpy as np
from sklearn.manifold import MDS

D = np.array([[0, 4, 5, 3], [4, 0, 3, 5],
              [5, 3, 0, 4], [3, 5, 4, 0]], float)
cases = {
    "defaults": dict(),
    "dissimilarity='precomputed'": dict(dissimilarity="precomputed",
                                        init="random"),
    "metric=True": dict(metric=True, init="random"),
    "explicit": dict(metric="precomputed", init="random",
                     n_init=1),
}
for name, kw in cases.items():
    X = D if "precomputed" in str(kw) else D[:, :2]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        MDS(n_components=2, random_state=0, **kw).fit(X)
    print(f"{name}: {len(rec)} warning(s)")
    for w in rec:
        msg = f"{w.category.__name__}: {w.message}"
        print(textwrap.fill(msg, 64, initial_indent="    ",
                            subsequent_indent="      "))

# What type_of_target says about seven kinds of target.
import numpy as np
from sklearn.datasets import load_linnerud
from sklearn.utils.multiclass import type_of_target

cases = [("two labels", ["ham", "spam", "ham"]),
         ("three classes", [0, 2, 1, 2]),
         ("0/1 label matrix", np.array([[1, 0, 1], [0, 1, 1]])),
         ("two class columns", [[0, 3], [1, 2], [2, 3]]),
         ("two real columns", [[1.5, 2.0], [0.3, 4.1]]),
         ("one real column", [0.5, 1.2, 3.3]),
         ("Linnerud targets", load_linnerud().target)]
for name, t in cases:
    print(f"{name:20s}{type_of_target(t)}")

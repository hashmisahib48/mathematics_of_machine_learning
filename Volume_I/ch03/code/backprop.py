# Forward and backward pass through the graph of a logistic loss.
import numpy as np

x, y = 1.5, 1.0                  # one sample
w, b = 0.8, -0.5                 # two parameters

u = w * x                                  # node 1: product
z = u + b                                  # node 2: sum
p = 1 / (1 + np.exp(-z))                   # node 3: sigmoid
L = -(y * np.log(p) + (1 - y) * np.log(1 - p))   # node 4
print(f"forward:  u = {u:.4f}  z = {z:.4f}  p = {p:.4f}"
      f"  L = {L:.4f}")

dL_dp = -y / p + (1 - y) / (1 - p)         # node 4, local
dp_dz = p * (1 - p)                        # node 3, local
dL_dz = dL_dp * dp_dz                      # Eq. 3.6-2
dL_db = dL_dz * 1.0                        # node 2 passes it on
dL_dw = dL_dz * x                          # node 1: du/dw = x
print(f"backward: dL/dp = {dL_dp:.4f}  dp/dz = {dp_dz:.4f}"
      f"  dL/dz = {dL_dz:.4f}")
print(f"          dL/dw = {dL_dw:.4f}  dL/db = {dL_db:.4f}")
print(f"p - y = {p - y:.4f}   (equals dL/dz)")

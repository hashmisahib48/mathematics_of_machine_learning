# Two facts about depth: linear pieces, and gradients at initialisation.
import numpy as np

relu = lambda z: np.maximum(z, 0)
def tent(x):                                # two ReLU units, Eq. 31.12-1
    return 2 * relu(x) - 4 * relu(x - 0.5)

def layer_gradients(act, depth=10, width=30, n=200, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 20); y = (rng.rand(n) > 0.5) * 1.0
    sizes = [20] + [width] * depth + [1]
    fac = 2 if act == "logistic" else 6     # as in _init_coef; b = 0
    Ws = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        k = np.sqrt(fac / (a + b))
        Ws.append(rng.uniform(-k, k, (a, b)))
    f, df = {"logistic": (lambda z: 1 / (1 + np.exp(-z)),
                          lambda a: a * (1 - a)),
             "tanh": (np.tanh, lambda a: 1 - a**2),
             "relu": (relu, lambda a: (a > 0) * 1.0)}[act]
    A = [X]
    for W in Ws[:-1]:
        A.append(f(A[-1] @ W))
    p = 1 / (1 + np.exp(-(A[-1] @ Ws[-1])[:, 0]))
    D = (p - y)[:, None]                    # Eq. 31.4-1
    rms = [np.sqrt(np.mean((A[-1].T @ D / n)**2))]
    for l in range(len(Ws) - 1, 0, -1):
        D = (D @ Ws[l].T) * df(A[l])        # Eq. 31.4-3
        rms.append(np.sqrt(np.mean((A[l - 1].T @ D / n)**2)))
    return rms[::-1]                        # layer 1 first


if __name__ == "__main__":
    x = np.linspace(0, 1, 2**12 + 1)            # grid: 2^12 + 1 points
    print("layers L   pieces of g^L   units: deep net   one layer")
    f = x.copy()
    for L in range(1, 9):
        f = tent(f)
        slope = np.diff(f) / np.diff(x)
        pieces = 1 + np.count_nonzero(np.abs(np.diff(slope)) > 1e-6)
        print(f"{L:8d}   {pieces:12d}   {2 * L:12d}   {2**L - 1:9d}")


    print("\nroot-mean-square gradient of the weights of each layer, 10 hidden")
    print("layers of 30 units, at initialisation (layer 1 is next to the input)")
    for act in ("logistic", "tanh", "relu"):
        g = layer_gradients(act)
        print(f"{act:9s}", " ".join(f"{v:.0e}" for v in g[:3]), "...",
              " ".join(f"{v:.0e}" for v in g[-2:]),
              f"  ratio first/last {g[0] / g[-1]:.1e}")

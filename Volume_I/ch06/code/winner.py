# Choosing the best of many useless models on the test set.
import numpy as np

rng = np.random.default_rng(64)
n_models, n_test, n_fresh = 200, 50, 20000

def one_round():
    y = rng.integers(0, 2, n_test)               # coin-flip labels
    X = rng.normal(size=(n_test, n_models))      # noise features
    # rule j: predict class 1 when feature j is positive
    acc = np.mean((X > 0) == y[:, None], axis=0)
    j = np.argmax(acc)                           # tune on test set
    y_new = rng.integers(0, 2, n_fresh)          # fresh data
    x_new = rng.normal(size=n_fresh)             # new draw
    return acc, acc[j], np.mean((x_new > 0) == y_new)

acc, best, fresh = one_round()
print(f"mean test accuracy of the 200 rules: {acc.mean():.3f}")
print(f"best test accuracy (the chosen rule): {best:.3f}")
print(f"accuracy of the chosen rule on fresh data: {fresh:.3f}")

rounds = [one_round()[1:] for _ in range(500)]   # repeat 500 times
b, f = np.mean(rounds, axis=0)
print(f"over 500 repetitions: best test {b:.3f}, fresh {f:.3f}")

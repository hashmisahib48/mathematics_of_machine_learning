# The invented "spectra" of Chapter 45: three groups of samples,
# each a mixture of five smooth curves, 10 000 channels, some noise.
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.datasets import make_blobs


def make_spectra(n=1000, p=10000, noise=0.5, seed=0):
    rng = np.random.RandomState(12345 + seed)
    Z, y = make_blobs(n_samples=n, n_features=5, centers=3,
                      cluster_std=2.5, random_state=seed)
    curves = gaussian_filter1d(rng.normal(size=(5, p)), 20, axis=1)
    curves /= curves.std()
    return Z @ curves + noise * rng.normal(size=(n, p)), y

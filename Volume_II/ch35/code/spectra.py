# Synthetic absorbance spectra: 100 channels, fat as the target.
import numpy as np

CH = np.arange(100)

def band(centre, width):
    return np.exp(-0.5 * ((CH - centre) / width) ** 2)

# pure-component spectra: absorbance per per cent of the component
FAT = 0.050 * band(30, 4) + 0.030 * band(72, 5)
PROT = 0.040 * band(26, 6) + 0.045 * band(60, 7)
WATER = 0.020 * band(48, 16) + 0.015 * band(85, 8)

def make_spectra(n, seed):
    rng = np.random.default_rng(seed)
    fat = 20 + 3 * rng.standard_normal(n)
    prot = 18 + 4 * rng.standard_normal(n)
    water = 100 - fat - prot + 3 * rng.standard_normal(n)
    u = np.linspace(-1, 1, 100)
    base = (rng.normal(0, 0.4, n)[:, None]
            + rng.normal(0, 0.3, n)[:, None] * u
            + rng.normal(0, 0.2, n)[:, None] * (u ** 2 - 1 / 3))
    X = (np.outer(fat, FAT) + np.outer(prot, PROT)
         + np.outer(water, WATER) + base)
    X = X * (1 + 0.08 * rng.standard_normal(n))[:, None]      # scatter
    X = X + 0.03 * rng.standard_normal((n, 100))              # noise
    y = fat + 0.5 * rng.standard_normal(n)       # reference assay
    return X, y

if __name__ == "__main__":
    X, y = make_spectra(60, 0)
    print("X", X.shape, "y", y.shape, " mean y %.2f, sd y %.2f"
          % (y.mean(), y.std(ddof=1)))

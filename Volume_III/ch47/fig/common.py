# Shared set-up of the Chapter 47 figure scripts.
import sys, os, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../../lib')
os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning)
warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
from scipy.stats import norm
from sklearn.neighbors import KernelDensity

plt.rcParams.update({'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.labelsize': 9,
                     'legend.fontsize': 8.5, 'axes.titlesize': 9})
NUM = {}


def waits(n, seed):
    rng = np.random.default_rng(seed)
    short = rng.random(n) < 0.35
    t = np.where(short, rng.normal(54, 5, n), rng.normal(80, 6, n))
    return t[:, None]


def truth(x):
    return 0.35 * norm.pdf(x, 54, 5) + 0.65 * norm.pdf(x, 80, 6)


X, X_test = waits(200, 1), waits(5000, 2)
XGRID = np.linspace(25, 115, 901)


def kde_curve(data, h, kernel='gaussian', grid=XGRID):
    k = KernelDensity(bandwidth=h, kernel=kernel).fit(data)
    return np.exp(k.score_samples(grid[:, None]))


def dump(name):
    json.dump(NUM, open('verify/' + name + '.json', 'w'), indent=1)

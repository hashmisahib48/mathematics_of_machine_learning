# Problem 51C.2: a column that is blank in 95 % of the rows. Writes fig/s51c2.png and prints the numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline

def make(kind, seed, n=2000):
    r = np.random.RandomState(seed); A = r.normal(size=(n, 3)); t = r.normal(size=n)
    u = A[:, 0] + 0.5 * t                                  # risk (depends on the test too)
    y = A @ [1.0, 0.5, 0.3] + 1.0 * t + 2.0 * (u > 1.4) + r.normal(size=n)       # test matters; also an alarm effect
    if kind == 'chance': blank = r.uniform(size=n) > 0.05
    else:                                                  # the test is ordered for the 5 % most alarming
        blank = u < np.quantile(u, 0.95)
    return A, np.where(blank, np.nan, t), y, blank, t
kf = KFold(5, shuffle=True, random_state=0)
rows = {}
for kind in ('chance', 'ordered when alarming'):
    out = {k: [] for k in ('column dropped', 'mean fill', 'mean fill + indicator', 'complete table (oracle)')}
    for s in range(12):
        A, t, y, blank, tfull = make('chance' if kind == 'chance' else 'x', s)
        base = LinearRegression()
        out['column dropped'].append(cross_val_score(base, A, y, cv=kf).mean())
        X = np.c_[A, t]
        out['mean fill'].append(cross_val_score(make_pipeline(SimpleImputer(), LinearRegression()), X, y, cv=kf).mean())
        out['mean fill + indicator'].append(cross_val_score(make_pipeline(SimpleImputer(add_indicator=True), LinearRegression()), X, y, cv=kf).mean())
        out['complete table (oracle)'].append(cross_val_score(LinearRegression(), np.c_[A, tfull], y, cv=kf).mean())
    rows[kind] = out
    print("--- column blank in 95 %% of the rows, %s: blank share %.3f" % (kind, blank.mean()))
    for k, v in out.items():
        print("   %-26s CV R2 %.3f +- %.3f (12 data sets)" % (k, np.nanmean(v), np.nanstd(v, ddof=1) / np.sqrt(12)))
fig, ax = plt.subplots(figsize=(4.4, 1.9), dpi=240)
names = ['column dropped', 'mean fill', 'mean fill + indicator']; w = 0.36
for i_, (kind, col) in enumerate((('chance', BLUE), ('ordered when alarming', OR))):
    vals = [np.mean(rows[kind][k]) for k in names]
    ax.bar(np.arange(3) + (i_ - 0.5) * w, vals, w, color=col, label='blank by chance' if i_ == 0 else 'ordered only when alarming')
    for x_, v_ in zip(np.arange(3) + (i_ - 0.5) * w, vals): ax.text(x_, v_ + 0.01, '%.2f' % v_, ha='center', fontsize=7.8, color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(['drop the column', 'mean fill', 'mean fill + indicator'], fontsize=8.3); ax.set_ylabel('cross-validated $R^2$', fontsize=8.5)
ax.tick_params(labelsize=8.3); ax.set_ylim(0, 0.9); ax.legend(fontsize=7.8, loc='upper left', ncol=2, handlelength=1.0, columnspacing=0.8); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s51c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s51c2.png') as im: w_ = im.size[0] / 240
print("figure width %.2f in, printed 4.4 in, smallest font %.1f pt -> printed %.1f pt" % (w_, fs, fs * 4.4 / w_))

# Problem 55C.2: the gap of TimeSeriesSplit. Writes fig/s55c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch55/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from cvdata import make_sales
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

# (a) residual autocorrelation of the book's figure: e_t = 0.9 e_{t-1} + noise
rng = np.random.RandomState(2); e = np.zeros(200000)
for k in range(1, len(e)): e[k] = 0.9 * e[k - 1] + rng.normal(0, 0.5)
lags = np.arange(0, 13); ac = np.array([np.corrcoef(e[:len(e) - h], e[h:])[0, 1] for h in lags])
print("autocorrelation of the residuals, lag 0..12:", ac.round(2).tolist())
print("first lag with 0.9^h below 0.05: %d" % next(h for h in range(1, 200) if 0.9 ** h < 0.05))
print("0.9^h for h = 1, 4, 5, 10: %s" % [round(0.9 ** h, 3) for h in (1, 4, 5, 10)])

# (b) RMSE of the forecast made d weeks ahead: TimeSeriesSplit(test_size=1, gap=d-1), last 30 weeks of 240 tested
ds = [1, 2, 4, 8, 16]; models = {"forest": RandomForestRegressor(10, random_state=0), "ridge": Ridge(1.0)}
R = {m: np.zeros(len(ds)) for m in models}; NS = 8
for seed in range(NS):
    X, y = make_sales(300, seed); X, y = X[:240], y[:240]
    for m, est in models.items():
        for j, d in enumerate(ds):
            cv = TimeSeriesSplit(30, test_size=1, gap=d - 1)
            R[m][j] += -cross_val_score(est, X, y, cv=cv, scoring="neg_mean_squared_error").mean() / NS
print("MSE of the d-weeks-ahead forecast (mean of %d series), d = %s" % (NS, ds))
for m in models: print("  %-7s RMSE %s" % (m, np.sqrt(R[m]).round(2).tolist()))
print("ratio of RMSE at d = 4 to d = 1: forest %.2f   ridge %.2f" % tuple(np.sqrt(R[m][2] / R[m][0]) for m in models))

fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9), dpi=240)
a = axs[0]; a.bar(lags, ac, color=BLUE, width=0.65); a.set_xlabel('lag h (weeks)'); a.set_ylabel('correlation'); a.set_ylim(0, 1.05); a.set_xticks([0, 1, 4, 8, 12])
a.text(1.9, 0.95, '%.2f' % ac[1], fontsize=8.2, color=NAVY); a.text(4.9, 0.7, '%.2f' % ac[4], fontsize=8.2, color=NAVY)
a = axs[1]; a.plot(ds, np.sqrt(R['forest']), '-o', color=RED, ms=3, lw=1.1, label='random forest'); a.plot(ds, np.sqrt(R['ridge']), '-s', color=GREEN, ms=3, lw=1.1, label='ridge')
a.set_xlabel('weeks ahead d = gap + 1'); a.set_ylabel('RMSE'); a.set_xticks(ds); a.set_ylim(0, 14); a.legend(fontsize=8.2, loc='lower right')
for a in axs: a.grid(color=GRID, lw=0.5); a.set_axisbelow(True); a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.6); a.yaxis.label.set_size(8.6)
fig.subplots_adjust(wspace=0.34)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s55c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s55c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

# Problem 35C.2: components chosen outside the loop. Writes fig/s35c2.png.
import sys, os, warnings
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch35/code'); os.chdir(HERE + '/..')
warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import KFold, cross_val_predict
from spectra import make_spectra
cv = KFold(5, shuffle=True, random_state=0)
def leaky(X, y, A): return rmse(y, cross_val_predict(LinearRegression(), PLSRegression(A).fit(X, y).x_scores_, y, cv=cv))
def honest(X, y, A): return rmse(y, cross_val_predict(PLSRegression(A), X, y, cv=cv).ravel())
rng = np.random.default_rng(0); Xr = rng.standard_normal((60, 100)); yr = rng.standard_normal(60)
print("pure noise: 60 samples, 100 features, sd of y %.3f; predicting the mean in cross-validation gives RMSE %.3f" % (yr.std(ddof=1), rmse(yr, cross_val_predict(LinearRegression(), np.zeros((60, 1)), yr, cv=cv))))
As = list(range(1, 11)); L = [leaky(Xr, yr, A) for A in As]; Hn = [honest(Xr, yr, A) for A in As]
print(" A  leaky CV RMSE  honest CV RMSE"); [print(f"{A:2d} {l:12.3f} {h:14.3f}") for A, l, h in zip(As, L, Hn)]
Xs, ys = make_spectra(60, 0); print("spectra, A = 10: leaky %.3f honest %.3f" % (leaky(Xs, ys, 10), honest(Xs, ys, 10)))
m = PLSRegression(10).fit(Xr, yr); T = m.x_scores_
print("training RMSE of PLSRegression(10) on the noise: %.3f, training R2 %.4f" % (rmse(yr, m.predict(Xr).ravel()), m.score(Xr, yr)))
print("in-sample correlation of the scores with y, k = 1..10:", np.round([np.corrcoef(T[:, k], yr)[0, 1] for k in range(10)], 3))
print("a single random direction: typical |corr| with y ~ 1/sqrt(n) = %.3f; best of 100 random directions: %.3f" % (1 / np.sqrt(60), np.mean([max(abs(np.corrcoef(Xr @ d, yr)[0, 1]) for d in np.random.default_rng(s).standard_normal((100, 100))) for s in range(20)])))
# the leak: the scores of the held-out samples were built with their own targets
tr, te = np.arange(48), np.arange(48, 60)
sc_all = PLSRegression(10).fit(Xr, yr).x_scores_
fit_in = LinearRegression().fit(sc_all[tr], yr[tr]); print("held-out 12 samples, scores from all 60: RMSE %.3f" % rmse(yr[te], fit_in.predict(sc_all[te])))
mm = PLSRegression(10).fit(Xr[tr], yr[tr]); print("held-out 12 samples, whole fit on the first 48: RMSE %.3f" % rmse(yr[te], mm.predict(Xr[te]).ravel()))
fig, ax = plt.subplots(figsize=(3.6, 2.0), dpi=240)
ax.plot(As, L, marker='o', ms=3.5, color=OR, lw=1.4, label='scores from all samples'); ax.plot(As, Hn, marker='s', ms=3.5, color=GREEN, lw=1.4, label='whole fit inside the loop')
ax.axhline(yr.std(ddof=1), color=GREY, ls='--', lw=1.0); ax.text(10.0, yr.std(ddof=1) - 0.05, 'sd of $y$ (the mean)', fontsize=9, color=GREY, va='top', ha='right')
ax.set_xlabel('number of components $A$'); ax.set_ylabel('cross-validated RMSE'); ax.set_ylim(0, 1.75); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(loc='lower right', fontsize=9, handlelength=1.4, bbox_to_anchor=(1.0, 0.18))
fig.subplots_adjust(left=0.14, right=0.98, top=0.97, bottom=0.2)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s35c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
w = Image.open('fig/s35c2.png').size[0] / 240; print("figure width %.2f in, printed 3.6 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 3.6 / w))

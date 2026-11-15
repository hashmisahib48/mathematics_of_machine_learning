# Problem 7C.3: one flower apart. Writes fig/s7c3.png, prints verify/c3.out
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from scipy.stats import binomtest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Eq. 7.9-1 for the two scores of the problem
m = 38; aA, aB = 37 / 38, 36 / 38
sA, sB = (math.sqrt(a * (1 - a) / m) for a in (aA, aB))
print("A = %.4f, B = %.4f, difference = %.4f (one flower = 1/38)" % (aA, aB, aA - aB))
print("s_A = %.4f, s_B = %.4f" % (sA, sB))
sd_diff = math.sqrt(sA ** 2 + sB ** 2)
print("independent test parts: s_diff = %.4f, difference / s_diff = %.2f" % (sd_diff, (aA - aB) / sd_diff))
abar = (aA + aB) / 2
need = 2 * abar * (1 - abar) / ((aA - aB) / 2) ** 2
print("test samples for the difference to be 2 s_diff: %.0f (iris has 150 flowers)" % need)
# on the same flowers only the flowers on which the models disagree matter: one flower, 1 against 0
print("sign test, A right and B wrong on 1 flower, B right and A wrong on 0: p = %.2f" % binomtest(1, 1, 0.5).pvalue)
print("sign test with 5 against 1: p = %.3f ; with 9 against 2: p = %.3f" % (binomtest(5, 6, 0.5).pvalue, binomtest(9, 11, 0.5).pvalue))

# --- a fairer comparison: the same 200 splits for both models, differences per split
X, y = load_iris(return_X_y=True)
dA, dB = [], []
for s in range(200):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s)
    dA.append(make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Xtr, ytr).score(Xte, yte))
    dB.append(make_pipeline(StandardScaler(), LogisticRegression(C=0.3)).fit(Xtr, ytr).score(Xte, yte))
dA, dB = np.array(dA), np.array(dB); d = dA - dB
print("\nmodel A: C = 1.0, model B: C = 0.3 (the same pipeline), 200 splits of 38 flowers")
print("A mean %.4f sd %.4f ; B mean %.4f sd %.4f" % (dA.mean(), dA.std(), dB.mean(), dB.std()))
print("A - B per split: mean %.4f, sd %.4f, min %.4f, max %.4f" % (d.mean(), d.std(ddof=1), d.min(), d.max()))
print("splits with A better: %d, equal: %d, B better: %d" % ((d > 1e-9).sum(), (abs(d) < 1e-9).sum(), (d < -1e-9).sum()))
print("splits where A - B = +1/38 exactly: %d" % (abs(d - 1 / 38) < 1e-9).sum())
print("standard error of the mean difference (200 overlapping splits, optimistic): %.4f" % (d.std(ddof=1) / math.sqrt(200)))

fig, axs = plt.subplots(1, 2, figsize=(5.2, 1.75), dpi=240, gridspec_kw={'wspace': 0.42})
ax = axs[0]; v, c = np.unique(dA.round(4), return_counts=True)
ax.bar(v, c, width=0.018, color=LB, ec=BLUE); ax.set_ylim(0, 82); ax.set_xlabel('accuracy of A, 200 splits', fontsize=8.5); ax.set_ylabel('splits', fontsize=8.5)
ax.axvline(aA, color=GREEN, lw=1.2); ax.axvline(aB, color=OR, lw=1.2, ls='--'); ax.text(aA + 0.004, ax.get_ylim()[1] * 0.97, 'A', color=GREEN, fontsize=8.5, va='top'); ax.text(aB - 0.004, ax.get_ylim()[1] * 0.97, 'B', color=OR, fontsize=8.5, va='top', ha='right')
ax = axs[1]; v, c = np.unique(d.round(4), return_counts=True)
ax.bar(v * 38, c, width=0.5, color=LB, ec=BLUE); ax.set_xlabel('flowers gained by A over B, per split', fontsize=8.5); ax.set_ylabel('splits', fontsize=8.5)
for a_ in axs: a_.tick_params(labelsize=8); a_.grid(axis='y', color=GRID, lw=0.5); a_.set_axisbelow(True)
save(fig, 'fig/s7c3.png')

# Problem 8C.2: the best of twenty coin-toss models. Writes fig/s8c2.png, prints verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
from figstyle import *
from scipy.stats import binom

n_te, n_mod = 50, 20
rng = np.random.default_rng(5)                       # the figure of the problem
sc = rng.binomial(n_te, 0.5, size=n_mod) / n_te
print("the figure of the problem: best %.2f, mean of the 20 %.3f, models above 0.60: %d" % (sc.max(), sc.mean(), (sc > 0.6).sum()))
print("P(one model scores >= 0.72) = %.5f ; P(at least one of 20 does) = %.4f" % (binom.sf(35, n_te, 0.5), 1 - binom.cdf(35, n_te, 0.5) ** n_mod))
print("P(best of 20 >= 0.60) = %.4f ; P(single model >= 0.60) = %.4f" % (1 - binom.cdf(29, n_te, 0.5) ** n_mod, binom.sf(29, n_te, 0.5)))
print("sd of one score sqrt(.25/50) = %.4f" % np.sqrt(0.25 / n_te))

R = 20000
rg = np.random.default_rng(1)
S = rg.binomial(n_te, 0.5, size=(R, n_mod)) / n_te   # test scores of the 20 models
best = S.max(1); pick = S.argmax(1)
fresh = rg.binomial(n_te, 0.5, size=R) / n_te        # winner scored on 50 new samples
print("%d repetitions: best of 20 on the same test part: mean %.3f (5th to 95th percentile %.2f to %.2f)" % (R, best.mean(), *np.percentile(best, [5, 95])))
print("the winner scored on 50 fresh samples: mean %.3f (5th to 95th percentile %.2f to %.2f)" % (fresh.mean(), *np.percentile(fresh, [5, 95])))
print("best of 20 reaches 0.62 or more in %.1f per cent of repetitions; a fresh score does in %.1f per cent" % (100 * (best >= 0.62).mean(), 100 * (fresh >= 0.62).mean()))
for m in (50, 200, 1000):
    e = rg.binomial(m, 0.5, size=(4000, n_mod)).max(1) / m
    print("test part of %4d samples: mean of the best of 20 = %.3f" % (m, e.mean()))

fig, ax = plt.subplots(figsize=(3.3, 1.75), dpi=240)
bins = np.arange(0.31, 0.80, 0.02)
ax.hist(best, bins=bins, color=LB, ec=BLUE, lw=0.7, label='best of 20, test part')
ax.hist(fresh, bins=bins, histtype='step', color=OR, lw=1.3, label='same winner, 50 new samples')
ax.axvline(0.5, color=GREY, lw=0.8, ls=':'); ax.set_xlabel('accuracy', fontsize=9); ax.set_ylabel('repetitions', fontsize=9)
ax.tick_params(labelsize=8); ax.legend(fontsize=7.5, loc='upper left', handlelength=1.2, borderaxespad=0.1)
ax.set_ylim(0, ax.get_ylim()[1] * 1.45); ax.set_yticks([])
fig.subplots_adjust(left=0.08, bottom=0.22); save(fig, 'fig/s8c2.png')

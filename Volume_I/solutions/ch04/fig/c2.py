# Problem 4C.2: the base rate. Writes fig/s4c2.png, verify/c2.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *

se, fp = 0.90, 0.05; LR = se / fp                    # likelihood ratio of one positive result
def post(prev, k):                                   # P(sick | k positive results), results independent given the status
    odds = prev / (1 - prev) * LR ** k; return odds / (1 + odds)
print("likelihood ratio of one positive result: %.1f" % LR)
print("prevalence  one +     two +     three +")
for pv in (0.0001, 0.001, 0.01, 0.1, 0.3, 0.5):
    print("%9.4f  %8.4f  %8.4f  %8.4f" % (pv, post(pv, 1), post(pv, 2), post(pv, 3)))
for k in (1, 2, 3):
    pstar = 1 / (1 + LR ** k); print("prevalence at which P(sick | %d +) = 0.5: %.4f" % (k, pstar))
print("posterior odds multiply by 18 per positive: 0.01 -> odds %.5f, x18 = %.4f, x324 = %.4f" % (0.01 / 0.99, 0.01 / 0.99 * 18, 0.01 / 0.99 * 324))
print("screening at 1 %%: %.4f   suspected patients at 30 %%: %.4f" % (post(0.01, 1), post(0.30, 1)))
print("two positives, 1 %%: %.4f   (book: 0.7660)" % post(0.01, 2))

fig, ax = plt.subplots(figsize=(4.4, 2.0), dpi=240)
pv = np.logspace(-4, np.log10(0.9), 300)
for k, col, ls in ((1, GREEN, '-'), (2, BLUE, '--'), (3, PUR, ':')):
    ax.plot(pv, post(pv, k), color=col, ls=ls, lw=1.5, label='%d positive test%s' % (k, '' if k == 1 else 's'))
ax.set_xscale('log'); ax.plot([0.01], [post(0.01, 1)], 'o', color=OR, ms=4.5); ax.plot([0.3], [post(0.3, 1)], 'o', color=RED, ms=4.5)
ax.text(0.0085, 0.2, 'screening, 1 %', fontsize=7.5, color=OR, ha='right'); ax.text(0.5, 0.56, 'suspected, 30 %', fontsize=7.5, color=RED, ha='center')
ax.set_xlabel('prevalence $P(D)$', fontsize=8); ax.set_ylabel('$P(D \\mid$ positive$)$', fontsize=8); ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.95); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); ax.set_ylim(0, 1.03)
save(fig, 'fig/s4c2.png')

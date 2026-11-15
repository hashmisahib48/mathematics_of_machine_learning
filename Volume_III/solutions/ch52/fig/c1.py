# Problem 52C.1: words never seen. Writes fig/s52c1.png and prints the numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch52/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from collections import Counter
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from corpus import messages

msgs, y = messages(); n = len(msgs); tok = CountVectorizer().build_analyzer(); T = [tok(m) for m in msgs]
allw = Counter(w for t in T for w in t); print("tokens %d, distinct words %d, words occurring in one message only %d (%.1f%% of the words, %.1f%% of the tokens)"
      % (sum(allw.values()), len(allw), sum(1 for w in allw if sum(w in set(t) for t in T) == 1), 100 * 134 / len(allw), 100 * 134 / sum(allw.values())))
rng = np.random.RandomState(0); sizes = [6, 12, 18, 24, 30, 36]; R = []
print("size  unseen  Good-Turing N1/N  seen in 1 training message  seen in >=2   (shares of the test tokens)")
for s_ in sizes:
    a = []
    for rep in range(40):
        idx = rng.permutation(n); tr, te = idx[:s_], idx[s_:]
        cnt = Counter(w for i in tr for w in T[i]); dfc = Counter(w for i in tr for w in set(T[i]))
        toks = [w for i in te for w in T[i]]
        un = np.mean([w not in cnt for w in toks]); h = np.mean([dfc.get(w, 0) == 1 for w in toks]); g = np.mean([dfc.get(w, 0) >= 2 for w in toks])
        N1 = sum(1 for w in cnt if cnt[w] == 1); gt = N1 / sum(cnt.values())
        a.append((un, gt, h, g))
    a = np.mean(a, axis=0); R.append(a); print("%4d  %.3f   %.3f             %.3f                       %.3f" % (s_, *a))
R = np.array(R)
print("at 36 training messages min_df=2 would turn %.3f of the test tokens from seen to dropped, so %.3f instead of %.3f of the tokens would be lost" % (R[-1, 2], R[-1, 0] + R[-1, 2], R[-1, 0]))
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.95), dpi=240)
a = axs[0]; a.plot(sizes, R[:, 0], '-o', color=RED, ms=3.5, lw=1.3, label='unseen tokens (observed)'); a.plot(sizes, R[:, 1], '--s', color=BLUE, ms=3.2, lw=1.1, label='Good–Turing $N_1/N$')
a.set_xlabel('training messages'); a.set_ylabel('share of test tokens'); a.set_ylim(0, 0.8); a.legend(fontsize=7.7, loc='upper right', handlelength=1.4); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
a = axs[1]; x = np.arange(len(sizes)); a.bar(x, R[:, 3], 0.65, color=GREEN, label='seen in $\\geq$ 2 messages'); a.bar(x, R[:, 2], 0.65, bottom=R[:, 3], color=OR, label='seen in 1 message'); a.bar(x, R[:, 0], 0.65, bottom=R[:, 2] + R[:, 3], color=RED, label='unseen')
a.set_xticks(x); a.set_xticklabels(sizes); a.set_xlabel('training messages'); a.set_ylim(0, 1.0); a.set_ylabel('share of test tokens'); a.legend(fontsize=7.7, loc='upper right', ncol=1, handlelength=1.0, bbox_to_anchor=(1.0, 1.0), framealpha=0.85, frameon=True, edgecolor='none')
a.set_ylim(0, 1.45); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a.grid(color=GRID, lw=0.5); a.set_axisbelow(True)
for a in axs: a.tick_params(labelsize=8.3); a.xaxis.label.set_size(8.5); a.yaxis.label.set_size(8.5)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s52c1.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s52c1.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))
print("of the test tokens that the 36-message vocabulary knows, the share that belongs to words seen in only one training message: %.3f" % (R[-1, 2] / (R[-1, 2] + R[-1, 3])))

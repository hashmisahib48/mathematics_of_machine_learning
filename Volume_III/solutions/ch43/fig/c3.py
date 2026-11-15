# Problem 43C.3: perplexity of LDA on training and held-out documents. Writes fig/s43c3.png, verify/c3.out
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch43/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from scipy.special import gammaln, digamma
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from corpus import make_docs

docs, hid = make_docs(0); Xc = CountVectorizer(stop_words='english', min_df=2).fit_transform(docs).toarray()
tr, te = Xc[:45], Xc[45:]; Ntr, Nte = tr.sum(), te.sum()
print("training tokens %d, held-out tokens %d, vocabulary %d" % (Ntr, Nte, Xc.shape[1]))

def beta_term(l):                                   # E[log p(beta|eta) - log q(beta|lambda)] as in the bound of scikit-learn
    lam = l.components_; eta = l.topic_word_prior_; V = lam.shape[1]
    E = digamma(lam) - digamma(lam.sum(1, keepdims=True))
    return np.sum((eta - lam) * E) + np.sum(gammaln(lam) - gammaln(eta)) + np.sum(gammaln(eta * V) - gammaln(lam.sum(1)))
def split(M, rng):                                  # random halves of the tokens of every document
    A = np.zeros_like(M); B = np.zeros_like(M)
    for i, row in enumerate(M):
        toks = np.repeat(np.arange(M.shape[1]), row.astype(int)); rng.shuffle(toks); h = len(toks) // 2
        np.add.at(A[i], toks[:h], 1); np.add.at(B[i], toks[h:], 1)
    return A, B

Ks = list(range(1, 9)); R = {k: [] for k in ('train', 'test', 'test_wo', 'compl', 'beta_per_token')}
for K in Ks:
    row = {k: [] for k in R}
    for sd in (0, 1, 2):
        l = LDA(K, max_iter=100, random_state=sd).fit(tr)
        p_tr, p_te = l.perplexity(tr), l.perplexity(te); bt = beta_term(l)
        row['train'].append(p_tr); row['test'].append(p_te); row['beta_per_token'].append(bt / Nte)
        row['test_wo'].append(np.exp(-(-Nte * np.log(p_te) - bt) / Nte))    # the same bound without the topic-word term
        phi = l.components_ / l.components_.sum(1, keepdims=True); sc = []
        for ss in range(5):
            A, B = split(te, np.random.RandomState(ss)); p = l.transform(A) @ phi
            sc.append(np.exp(-(B * np.log(p)).sum() / B.sum()))
        row['compl'].append(np.mean(sc))
    for k in R: R[k].append(float(np.mean(row[k])))
print("\n K   training  held-out(sklearn)  topic-word term per token  held-out without it  document completion")
for i, K in enumerate(Ks):
    print(f"{K:2d}{R['train'][i]:11.1f}{R['test'][i]:15.1f}{R['beta_per_token'][i]:21.2f}{R['test_wo'][i]:21.1f}{R['compl'][i]:18.1f}")

fig, ax = plt.subplots(figsize=(5.4, 1.95), dpi=240)
ax.plot(Ks, R['train'], '-o', color=BLUE, ms=3.2, lw=1.3, label='training'); ax.plot(Ks, R['test'], '--s', color=RED, ms=3.2, lw=1.3, label='held out, scikit-learn')
ax.plot(Ks, R['test_wo'], '-.^', color=OR, ms=3.4, lw=1.3, label='held out, without the topic-word term'); ax.plot(Ks, R['compl'], '-D', color=GREEN, ms=3.0, lw=1.4, label='held out, document completion')
ax.set_xlabel('topics K', fontsize=8.6); ax.set_ylabel('perplexity', fontsize=8.6); ax.set_xticks(Ks); ax.tick_params(labelsize=8.6); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc='lower left', fontsize=8.6, handlelength=1.5, borderaxespad=0.2, labelspacing=0.25, bbox_to_anchor=(1.01, 0.0))
fig.subplots_adjust(left=0.1, right=0.6, top=0.97, bottom=0.22)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s43c3.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s43c3.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.9 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.9 / w))

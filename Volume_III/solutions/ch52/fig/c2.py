# Problem 52C.2: stop words, counts and tf-idf with naive Bayes. Writes fig/s52c2.png and prints the numbers.
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../lib'); sys.path.insert(0, HERE + '/../../../ch52/code'); os.chdir(HERE + '/..')
import warnings; warnings.simplefilter('error', FutureWarning); warnings.simplefilter('error', DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from corpus import messages

msgs, y = messages(); y = np.array(y); n = len(msgs)
cvr = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=0)
variants = {'counts': CountVectorizer(), 'counts, stop words out': CountVectorizer(stop_words='english'),
            'tf-idf': TfidfVectorizer(), 'tf-idf, stop words out': TfidfVectorizer(stop_words='english')}
fold = {k: [] for k in variants}; wrong = {k: [] for k in variants}; marg = {k: [] for k in variants}
for tr, te in cvr.split(msgs, y):
    for k, v in variants.items():
        vec = clone(v).fit([msgs[i] for i in tr]); Xa = vec.transform([msgs[i] for i in tr]); Xb = vec.transform([msgs[i] for i in te])
        nb = MultinomialNB().fit(Xa, y[tr]); p = nb.predict(Xb); fold[k].append((p == y[te]).mean())
        for i, ok in zip(te, p == y[te]):
            if not ok: wrong[k].append(i)
        if k == 'counts':                                   # split the log-odds of the true class against the best wrong class into stop-word and content parts
            words = vec.get_feature_names_out(); stop = np.array([w in ENGLISH_STOP_WORDS for w in words]); L = nb.feature_log_prob_
            for r_, i in enumerate(te):
                x = Xb[r_].toarray()[0]; sc = nb.class_log_prior_ + L @ x; t = y[i]; o = np.argmax(np.where(np.arange(4) == t, -np.inf, sc))
                d = L[t] - L[o]; marg[k].append((p[r_] == t, (x * d)[stop].sum(), (x * d)[~stop].sum(), nb.class_log_prior_[t] - nb.class_log_prior_[o]))
for k in variants: print("%-24s mean %.4f  sd %.4f   mistakes in 20 folds: %d of 240" % (k, np.mean(fold[k]), np.std(fold[k]), len(wrong[k])))
print("fold accuracies of 'counts, stop words out' and 'tf-idf, stop words out' identical:", np.allclose(fold['counts, stop words out'], fold['tf-idf, stop words out']),
      "; mistakes identical:", sorted(wrong['counts, stop words out']) == sorted(wrong['tf-idf, stop words out']))
print("fold accuracies of 'tf-idf' and 'tf-idf, stop words out' identical:", np.allclose(fold['tf-idf'], fold['tf-idf, stop words out']))
cvv = CountVectorizer().fit(msgs); words = cvv.get_feature_names_out(); stop = np.array([w in ENGLISH_STOP_WORDS for w in words]); X = cvv.transform(msgs).toarray().astype(float)
print("vocabulary %d words, of which %d are in the stop list; stop-word tokens %.3f of all tokens (%d of %d)" % (len(words), stop.sum(), X[:, stop].sum() / X.sum(), X[:, stop].sum(), X.sum()))
idf = np.log((1 + n) / (1 + (X > 0).sum(0))) + 1; W = X * idf
print("share of the total weight carried by stop words: counts %.3f   tf-idf (before unit length) %.3f   after L2: share of squared length %.3f" % (X[:, stop].sum() / X.sum(), W[:, stop].sum() / W.sum(), ((W / np.linalg.norm(W, axis=1, keepdims=True)) ** 2)[:, stop].sum() / n))
print("mean idf of stop words (by token) %.3f, of content words %.3f" % ((X[:, stop] * idf[stop]).sum() / X[:, stop].sum(), (X[:, ~stop] * idf[~stop]).sum() / X[:, ~stop].sum()))
M = np.array(marg['counts'], float); ok = M[:, 0] == 1
print("counts + naive Bayes with stop words, log-odds of the true class over the best other class (mean over test messages):")
for name, sel in (("correctly classified (%d)" % ok.sum(), ok), ("misclassified (%d)" % (~ok).sum(), ~ok)):
    print("   %-26s stop words %+.2f   content words %+.2f   priors %+.2f" % (name, M[sel, 1].mean(), M[sel, 2].mean(), M[sel, 3].mean()))
print("standard deviation over the test messages of the stop-word part %.2f, of the content part %.2f" % (M[:, 1].std(), M[:, 2].std()))
fig, ax = plt.subplots(figsize=(4.4, 1.85), dpi=240); names = list(variants); lab = ['counts', 'counts\nstop out', 'tf-idf', 'tf-idf\nstop out']
cnt = [len(wrong[k]) for k in names]; ax.bar(range(4), cnt, 0.6, color=[RED, GREEN, OR, GREEN])
for i_, v_ in enumerate(cnt): ax.text(i_, v_ + 0.5, '%d' % v_, ha='center', fontsize=8.3, color=NAVY)
ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8.3); ax.set_ylabel('mistakes in 240 tests', fontsize=8.5); ax.tick_params(labelsize=8.3); ax.set_ylim(0, max(cnt) * 1.2); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(pad=0.4)
fs = min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('fig/s52c2.png', dpi=240, bbox_inches='tight', pad_inches=0.04, facecolor='white')
with Image.open('fig/s52c2.png') as im: w = im.size[0] / 240
print("figure width %.2f in, printed 4.4 in, smallest font %.1f pt -> printed %.1f pt" % (w, fs, fs * 4.4 / w))
# which ingredient of tf-idf can change the decisions of naive Bayes once the stop words are out?
print("--- stop words out; the two ingredients of tf-idf separately (mistakes in 20 folds)")
for name, vec in (('counts', CountVectorizer(stop_words='english')), ('unit rows only (no idf)', TfidfVectorizer(stop_words='english', use_idf=False)),
                  ('idf only (no unit length)', TfidfVectorizer(stop_words='english', norm=None)), ('both (tf-idf)', TfidfVectorizer(stop_words='english'))):
    wr = []
    for tr, te in cvr.split(msgs, y):
        v = clone(vec).fit([msgs[i] for i in tr]); nb = MultinomialNB().fit(v.transform([msgs[i] for i in tr]), y[tr]); p = nb.predict(v.transform([msgs[i] for i in te])); wr += [i for i, ok in zip(te, p == y[te]) if not ok]
    print("   %-26s %d mistakes, messages %s" % (name, len(wr), sorted(set(int(i) for i in wr))))
print("class counts in a training fold:", np.bincount(y[next(cvr.split(msgs, y))[0]]).tolist(), "(equal priors)")
print("message that every stop-words-out model gets wrong:", repr(msgs[4]), "topic", int(y[4]))

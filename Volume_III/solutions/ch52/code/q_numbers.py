# Numbers quoted in the model answers to the questions of Chapter 52
import os, sys, warnings
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.append(HERE + "/../../../ch52/code")
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from corpus import messages, TOPIC_NAMES
msgs, y = messages(); y = np.array(y); n = len(msgs)

print("=== Question 1: idf = ln(n/df) against the library's ln((1+n)/(1+df)) + 1")
cv = CountVectorizer(); X = cv.fit_transform(msgs); df = np.asarray((X > 0).sum(0)).ravel(); w = cv.get_feature_names_out()
j = list(w).index("the"); print("n = %d, df(the) = %d, df=1 words: %d" % (n, df[j], (df == 1).sum()))
lib = lambda d, n: np.log((1 + n) / (1 + d)) + 1; txt = lambda d, n: np.log(n / d)
print("the:   library %.4f   textbook %.4f   ratio %.1f" % (lib(df[j], n), txt(df[j], n), lib(df[j], n) / txt(df[j], n)))
print("df=1:  library %.4f   textbook %.4f" % (lib(1, n), txt(1, n)))
print("df=n:  library %.4f   textbook %.4f" % (lib(n, n), txt(n, n)))
s = "the chef stirred the tomato sauce in a hot pan"
for name, f in (("library", lib), ("textbook", txt)):
    c = cv.transform([s]).toarray()[0]; idf = f(df, n); v = c * idf; v = v / np.linalg.norm(v)
    print("%-8s unit row of the sentence: weight of 'the' %.3f, largest content weight %.3f (%s)" % (name, v[j], np.max(np.delete(v, j)), w[np.argmax(np.where(np.arange(len(w)) == j, -1, v))]))
tiny = ["the cat", "the dog", "the"]; c3 = CountVectorizer().fit_transform(tiny).toarray(); d3 = (c3 > 0).sum(0)
print("three messages", tiny, "df", d3.tolist(), "-> textbook idf", txt(d3, 3).round(3).tolist(), "-> row 3 (message 'the') weights", (c3[2] * txt(d3, 3)).round(3).tolist(), " library:", (c3[2] * lib(d3, 3)).round(3).tolist())

print("\n=== Question 2: collisions for a vocabulary of one million words")
V = 1_000_000
print("%10s %8s %22s %26s" % ("m", "m/V", "words with a neighbour", "words without own bucket"))
for e in (20, 22, 24, 26, 28):
    m = 2 ** e; share = 1 - (1 - 1 / m) ** (V - 1); lost = V - m * (1 - (1 - 1 / m) ** V)
    print("2^%-8d %8.1f %21.4f %14.0f  (%.4f of V)" % (e, m / V, share, lost, lost / V))
print("coefficients of a linear model per class with m = 2^24: %d bytes (float64)" % (2 ** 24 * 8))
print("small corpus: V = 216 words, m = 2^12 = 4096 -> m/V = %.1f, share with a neighbour %.3f" % (4096 / 216, 1 - (1 - 1 / 4096) ** 215))
print("mean error of the cosine between the 1128 pairs of messages (unit-length count rows):")
C = CountVectorizer().fit_transform(msgs).astype(float); C = C.multiply(1 / np.sqrt(C.multiply(C).sum(1))).tocsr(); true = (C @ C.T).toarray(); iu = np.triu_indices(n, 1)
for m in (8, 64, 1024):
    row = []
    for alt in (False, True):
        Z = HashingVectorizer(n_features=m, norm="l2", alternate_sign=alt).transform(msgs); row.append(((Z @ Z.T).toarray()[iu] - true[iu]).mean())
    print("   m = %4d   unsigned %+.4f   signed %+.4f" % (m, *row))

print("\n=== Question 3: a message with no known word")
tf = TfidfVectorizer(); clf = LogisticRegression(C=10, max_iter=1000); pipe = make_pipeline(tf, clf).fit(msgs, y)
for s in ("zebra quantum xylophone", "the", "zebra quantum garlic", "garlic"):
    Z = tf.transform([s]); p = pipe.predict_proba([s])[0]
    print("%-26s non-zero cells %d  row length %.3f  predicted %-10s probabilities %s" % (repr(s), Z.nnz, np.sqrt(Z.multiply(Z).sum()), TOPIC_NAMES[int(np.argmax(p))], p.round(3).tolist()))
print("intercepts of the classifier", clf.intercept_.round(3).tolist(), " softmax of the intercepts", (np.exp(clf.intercept_) / np.exp(clf.intercept_).sum()).round(3).tolist())
nb = make_pipeline(CountVectorizer(), MultinomialNB()).fit(msgs, y); print("naive Bayes on 'zebra quantum xylophone': probabilities", nb.predict_proba(["zebra quantum xylophone"])[0].round(3).tolist(), "(the priors)")
cov = lambda s: np.mean([t in tf.vocabulary_ for t in tf.build_analyzer()(s)])
print("share of tokens in the vocabulary:", {s: round(float(cov(s)), 2) for s in ("zebra quantum xylophone", "the", "zebra quantum garlic")})

print("\n=== Question 4: when is min_df = 2 right? (data made in code from the four topics of the corpus)")
tok = CountVectorizer().build_analyzer()
topic_words = [[t for s, k in zip(msgs, y) if k == c for t in tok(s)] for c in range(4)]
def draw(nm, seed, typos=3, length=10):
    r = np.random.RandomState(seed); out, lab = [], []
    for i in range(nm):
        c = i % 4; words = list(r.choice(topic_words[c], length)) + ["typo%d" % t for t in r.randint(0, 5000, typos)]
        r.shuffle(words); out.append(" ".join(words)); lab.append(c)
    return out, np.array(lab)
print("%6s %10s %10s %10s %10s" % ("n", "min_df=1", "min_df=2", "min_df=5", "words kept (1, 2)"))
for nm in (48, 400, 4000):
    acc = {1: [], 2: [], 5: []}; kept = {1: [], 2: []}
    for rep in range(5):
        D, L = draw(nm, rep)
        for md in acc:
            acc[md].append(cross_val_score(make_pipeline(TfidfVectorizer(min_df=md), LogisticRegression(C=10, max_iter=1000)), D, L, cv=StratifiedKFold(4, shuffle=True, random_state=rep)).mean())
        for md in kept: kept[md].append(len(TfidfVectorizer(min_df=md).fit(D).vocabulary_))
    print("%6d %10.4f %10.4f %10.4f %12.0f, %.0f" % (nm, np.mean(acc[1]), np.mean(acc[2]), np.mean(acc[5]), np.mean(kept[1]), np.mean(kept[2])))

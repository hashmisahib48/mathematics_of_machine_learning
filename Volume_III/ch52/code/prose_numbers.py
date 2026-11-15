# Numbers quoted in the prose of Chapter 52 (printed and saved).
import json, os
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer,
    TfidfVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS)
from sklearn.preprocessing import normalize
from corpus import messages, make_docs

N = {}
msgs, y = messages(); y = np.array(y)
cv = CountVectorizer(); X = cv.fit_transform(msgs)
n, V = X.shape
tok = X.sum()
N["corpus"] = dict(n=n, V=V, nnz=int(X.nnz), tokens=int(tok),
    share=round(X.nnz / (n * V), 4),
    words_per_message=[min(len(m.split()) for m in msgs),
                       max(len(m.split()) for m in msgs)],
    dense_bytes=n * V * 8,
    csr_bytes=int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes))
df = np.asarray((X > 0).sum(axis=0)).ravel()
w = cv.get_feature_names_out()
N["df"] = dict(df1=int((df == 1).sum()), df2=int((df == 2).sum()),
    max_df=int(df.max()), max_word=str(w[df.argmax()]),
    top5=[(str(w[j]), int(df[j])) for j in np.argsort(-df)[:5]],
    stop_list=len(ENGLISH_STOP_WORDS))
N["idf"] = {str(d): round(float(np.log((1 + n) / (1 + d)) + 1), 4)
            for d in (1, 2, 5, 10, 30, 48)}
N["idf_plain"] = {str(d): round(float(np.log(n / d) + 1), 4)
                  for d in (1, 2, 5, 10, 30, 48)}
N["sublinear"] = {str(t): round(1 + np.log(t), 4) for t in (1, 2, 5, 10)}
# stop words and the 'the' weight
sw = CountVectorizer(stop_words="english").fit(msgs)
N["stop_vocab"] = len(sw.vocabulary_)
N["the_df"] = int(df[list(w).index("the")])
# min_df=2: messages that lose every word
m2 = CountVectorizer(min_df=2).fit(msgs); Xm = m2.transform(msgs)
N["min_df2"] = dict(V=len(m2.vocabulary_),
    empty_rows=int((np.asarray(Xm.sum(axis=1)).ravel() == 0).sum()),
    min_words=int(np.asarray(Xm.sum(axis=1)).ravel().min()))
for r in ((1, 2), (1, 3), (2, 2)):
    N["ngram%d%d" % r] = len(CountVectorizer(ngram_range=r).fit(msgs).vocabulary_)
N["char_wb24"] = len(CountVectorizer(analyzer="char_wb",
                     ngram_range=(2, 4)).fit(msgs).vocabulary_)
# tokenisation examples
text = "Don't stir the café sauce; 2 eggs, e-mail me at 9am!"
N["tokens_default"] = CountVectorizer().build_analyzer()(text)
N["tokens_ascii"] = CountVectorizer(strip_accents="unicode").build_analyzer()(text)
N["tokens_bigram"] = CountVectorizer(ngram_range=(1, 2)).build_analyzer()("stir the soup")
N["tokens_char_wb"] = CountVectorizer(analyzer="char_wb",
    ngram_range=(3, 3)).build_analyzer()("hot pan")
N["tokens_stop"] = CountVectorizer(stop_words="english",
    ngram_range=(2, 2)).build_analyzer()("stir the soup and taste it")
# cosine and distance identity
tv = TfidfVectorizer(); Z = tv.fit_transform(msgs)
a, b = Z[0].toarray().ravel(), Z[5].toarray().ravel()
N["identity"] = dict(d2=float(((a - b) ** 2).sum()), two_one_minus_cos=float(2 * (1 - a @ b)))
# the weights of one message
j = 0; row = X[j].toarray().ravel(); zr = Z[j].toarray().ravel()
N["message0"] = dict(text=msgs[0], counts={str(w[k]): int(row[k]) for k in np.nonzero(row)[0]},
    tfidf={str(tv.get_feature_names_out()[k]): round(float(zr[k]), 3) for k in np.nonzero(zr)[0]})
# toy example
docs = ["bake the bread", "bake the cake bake", "cake and the cake"]
Xt = CountVectorizer().fit_transform(docs).toarray(); dft = (Xt > 0).sum(0)
idft = np.log(4 / (1 + dft)) + 1; Wt = Xt * idft
N["toy"] = dict(norms=np.linalg.norm(Wt, axis=1).round(4).tolist(),
    idf=idft.round(4).tolist(), W=Wt.round(4).tolist(),
    plain_idf=(np.log(3 / dft) + 1).round(4).tolist())
Zt = normalize(Wt); N["toy"]["cos"] = (Zt @ Zt.T).round(4).tolist()
Zc = normalize(Xt.astype(float)); N["toy"]["cos_counts"] = (Zc @ Zc.T).round(4).tolist()
N["toy"]["dist_12"] = float(np.linalg.norm(Zt[0] - Zt[1]))
# 60 documents
docs60, lab = make_docs(0)
X60 = CountVectorizer(stop_words="english").fit_transform(docs60)
N["docs60"] = dict(shape=list(X60.shape), tokens=int(X60.sum()))
print(json.dumps(N, indent=1, default=str))
json.dump(N, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify", "prose_numbers.json"), "w"), indent=1, default=str)
# messages left without any word by min_df
extra = {}
for k in (2, 3, 4):
    Xk = CountVectorizer(min_df=k).fit(msgs).transform(msgs)
    extra[k] = int((np.asarray(Xk.sum(axis=1)).ravel() == 0).sum())
print("empty rows by min_df:", extra)
json.dump(extra, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verify", "empty_rows.json"), "w"))

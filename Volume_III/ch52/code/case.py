# Section 52.11: sorting the 48 messages by topic.
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RepeatedStratifiedKFold,
                                     cross_val_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from corpus import messages, TOPIC_NAMES

msgs, y = messages()
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=5,
                             random_state=0)
LR = lambda: LogisticRegression(C=10, max_iter=1000)
tf = lambda **kw: TfidfVectorizer(**kw)
models = {
    "counts + NB": make_pipeline(CountVectorizer(),
                                 MultinomialNB()),
    "tf-idf + NB": make_pipeline(tf(), MultinomialNB()),
    "tf-idf + LR": make_pipeline(tf(), LR()),
    "stop words + LR": make_pipeline(
        tf(stop_words="english"), LR()),
}
for name, p in models.items():
    s = cross_val_score(p, msgs, y, cv=cv)  # refitted per fold
    v = len(p[0].fit(msgs).vocabulary_)         # words on all 48
    print(f"{name:16s} accuracy {s.mean():.4f} (sd {s.std():.3f})"
          f"  words {v}")

best = make_pipeline(tf(stop_words="english"), LR()).fit(msgs, y)
words = best[0].get_feature_names_out()
for k, topic in enumerate(TOPIC_NAMES):         # largest weights
    top = np.argsort(-best[1].coef_[k])[:5]
    print(f"{topic:10s}", ", ".join(words[top]))
new = ["the striker hit the post", "bake bread with butter",
       "the telescope found a new planet",
       "rates and loans at the bank"]
print([TOPIC_NAMES[k] for k in best.predict(new)])

# Section 52.9: hashing words and dictionaries into columns.
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer)
from sklearn.utils import murmurhash3_32
from corpus import messages

hv = HashingVectorizer(n_features=16, norm=None)    # no fit needed
for w in ["cake"]:
    h = murmurhash3_32(w, seed=0, positive=False)   # signed 32 bits
    j, s = abs(h) % 16, 1 if h >= 0 else -1         # Eq. 52.5-1
    X = hv.transform([w])
    print(w, h, "-> bucket", j, "sign", s, X.indices, X.data)
    assert X.indices[0] == j and X.data[0] == s

msgs, _ = messages()
big = HashingVectorizer(n_features=2**20, norm=None,
                        alternate_sign=False).transform(msgs)
vocab = CountVectorizer().fit_transform(msgs)
print("hashed", big.shape, "nnz", big.nnz, "| vocabulary",
      vocab.shape, "nnz", vocab.nnz)

rows = [{"city": "Ipoh", "rooms": 3, "area": 80.0},
        {"city": "Penang", "rooms": 2, "area": 55.5}]
dv = DictVectorizer(sparse=False)
print(dv.fit_transform(rows).tolist(), list(dv.feature_names_))
fh = FeatureHasher(n_features=8, alternate_sign=False)
print(fh.transform(rows).toarray().tolist())

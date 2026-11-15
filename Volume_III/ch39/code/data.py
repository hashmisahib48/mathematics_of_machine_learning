# Data of the chapter: nested rings, a six-point toy, an invented corpus.
import numpy as np
from sklearn.datasets import make_circles


def rings(n=300, seed=0):
    """Two rings of lamp posts, one inside the other (Section 39.1)."""
    return make_circles(n_samples=n, factor=0.4, noise=0.05, random_state=seed)


# six points in two triangles joined by one short link (Section 39.2)
TOY = np.array([[0, 0], [1, 2], [2, 0], [5, 0], [6, 2], [7, 0]], float)


def toy_adjacency(radius=3.0):
    """Join two points when their distance is at most the radius."""
    d = np.sqrt(((TOY[:, None] - TOY[None]) ** 2).sum(-1))
    return ((d <= radius) & (d > 0)).astype(float)


# ---- an invented corpus: 80 sentences on four topics, made from templates
TOPICS = ['cooking', 'football', 'astronomy', 'finance']
VOC = {
 'cooking': dict(
   n=['onion', 'garlic', 'dough', 'butter', 'sauce', 'pasta', 'soup',
      'oven', 'recipe', 'pepper', 'flour', 'pan'],
   v=['simmer', 'chop', 'bake', 'stir', 'roast', 'whisk', 'fry', 'boil'],
   a=['fresh', 'warm', 'crispy', 'salty', 'golden', 'tender'],
   bridge=['season', 'market']),
 'football': dict(
   n=['striker', 'goalkeeper', 'referee', 'penalty', 'league', 'coach',
      'midfielder', 'stadium', 'defender', 'goal', 'captain', 'trophy'],
   v=['scored', 'saved', 'tackled', 'passed', 'won', 'missed', 'chased',
      'cheered'],
   a=['late', 'dramatic', 'narrow', 'home', 'unbeaten', 'noisy'],
   bridge=['star', 'team', 'season']),
 'astronomy': dict(
   n=['telescope', 'galaxy', 'planet', 'comet', 'orbit', 'nebula', 'moon',
      'asteroid', 'eclipse', 'satellite', 'sun', 'cosmos'],
   v=['orbits', 'shines', 'appears', 'crosses', 'reveals', 'glows', 'drifts',
      'fades'],
   a=['distant', 'bright', 'faint', 'massive', 'cold', 'ancient'],
   bridge=['star', 'team']),
 'finance': dict(
   n=['bank', 'stock', 'bond', 'interest', 'inflation', 'dividend',
      'investor', 'loan', 'profit', 'currency', 'budget', 'mortgage'],
   v=['rose', 'fell', 'paid', 'raised', 'cut', 'traded', 'borrowed',
      'invested'],
   a=['quarterly', 'steady', 'sharp', 'global', 'annual', 'cautious'],
   bridge=['market', 'team', 'season'])}
STRAY = 0.25
TEMPLATES = ['the {a} {n1} {v} the {n2}', 'a {n1} and a {n2} {v} in the {a} {n3}',
             'every {a} {n1} {v} beside the {n2}', 'the {n1} of the {n2} {v} {a}']


def _sentence(rng, t):
    V = VOC[TOPICS[t]]
    tmpl = TEMPLATES[rng.integers(len(TEMPLATES))]
    nn = list(rng.choice(V['n'], 3, replace=False))
    if rng.random() < 0.35:                          # a word that topics share
        nn[rng.integers(1, 3)] = rng.choice(V['bridge'])
    if rng.random() < STRAY:                         # a word from another topic
        other = VOC[TOPICS[(t + 1 + rng.integers(3)) % 4]]
        nn[rng.integers(0, 3)] = rng.choice(other['n'])
    return tmpl.format(a=rng.choice(V['a']), v=rng.choice(V['v']),
                       n1=nn[0], n2=nn[1], n3=nn[2])


def corpus(per_topic=20, seed=5, sentences=2):
    """Return (documents, topic of each document, vocabulary of each topic).
    A document is `sentences` invented sentences on one topic."""
    rng = np.random.default_rng(seed)
    docs, topic = [], []
    for t in range(4):
        for _ in range(per_topic):
            docs.append('. '.join(_sentence(rng, t) for _ in range(sentences)))
            topic.append(t)
    order = rng.permutation(len(docs))               # shuffle the documents
    docs = [docs[k] for k in order]
    topic = np.array(topic)[order]
    truth = {name: set(sum((VOC[name][k] for k in 'nva'), [])) for name in TOPICS}
    return docs, topic, truth

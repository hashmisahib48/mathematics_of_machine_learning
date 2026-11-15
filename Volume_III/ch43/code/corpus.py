# An invented corpus for Chapter 43: 48 short sentences on four topics
# (cooking, football, astronomy, finance) from which make_docs builds
# 60 documents, twelve of them mixing two topics. It is made up for
# this book and is not real text.
COOKING = [
    "the chef stirred the tomato sauce in a hot pan",
    "bake the bread in a hot oven until the crust is golden",
    "add garlic and onion to the butter and simmer the soup",
    "knead the dough with flour and salt and let it rest",
    "the recipe asks for fresh pasta, cheese and black pepper",
    "roast the onion and garlic with salt and a little butter",
    "a pinch of spice and a spoon of butter improve the sauce",
    "the kitchen smelled of warm bread and roast garlic",
    "stir the soup and taste it before you add more salt",
    "the chef chopped the onion and tomato for the pasta sauce",
    "mix flour and butter into a smooth dough and bake it",
    "season the roast with pepper, garlic and a pinch of spice",
]
FOOTBALL = [
    "the striker scored a late goal and the stadium roared",
    "the keeper saved a penalty in the final match of the season",
    "the coach praised the defender after the cup match",
    "fans filled the stadium to watch the team win the league",
    "the referee gave a penalty after a tackle in midfield",
    "a corner kick, a header and the striker scored again",
    "the team lost the match but the keeper played well",
    "the league season ends with the cup final at the stadium",
    "the midfield controlled the pitch and the team scored twice",
    "the coach changed the defender and the striker at half time",
    "the crowd cheered each tackle and every goal of the match",
    "the cup winners lifted the trophy while the fans sang",
]
ASTRONOMY = [
    "the telescope showed a bright comet near the moon tonight",
    "the planet orbits its star once every few hundred days",
    "a supernova can outshine a whole galaxy for weeks",
    "the observatory recorded the light of a distant galaxy",
    "mars and jupiter shine in the night sky near the moon",
    "an eclipse hid the sun while the comet crossed the sky",
    "the black hole bends the light of the star behind it",
    "astronomers used the telescope to count the planets of the star",
    "the nebula glows where a young star heats the cosmic dust",
    "an asteroid passed the moon and left the planet unharmed",
    "the universe holds billions of galaxy clusters and cosmic dust",
    "the orbit of the comet brings it close to the sun",
]
FINANCE = [
    "the bank raised the interest rate to fight inflation",
    "investors sold shares as the stock market fell sharply",
    "the fund bought bonds and paid a dividend to investors",
    "a loan with a low interest rate helps the budget",
    "the market expects inflation and higher interest rates",
    "the company reported a profit and its shares rose",
    "the bank approved the loan and the credit improved",
    "tax changes and a tight budget slowed the economy",
    "the price of the stock fell after the profit warning",
    "investors trade bonds and shares when interest rates move",
    "the economy grew while the bank held the interest rate",
    "the fund reported a strong profit and raised its dividend",
]
SENTENCES = [COOKING, FOOTBALL, ASTRONOMY, FINANCE]
TOPIC_NAMES = ["cooking", "football", "astronomy", "finance"]


def make_docs(seed=0, per_topic=12, n_mixed=12):
    """Documents of three sentences from one topic, plus mixed ones
    of two sentences from each of two topics. Returns (docs, labels);
    the label of a mixed document is -1."""
    import numpy as np
    rng = np.random.RandomState(seed)
    docs, lab = [], []
    for t, sents in enumerate(SENTENCES):
        for _ in range(per_topic):
            pick = rng.choice(len(sents), 3, replace=False)
            docs.append(" . ".join(sents[j] for j in pick))
            lab.append(t)
    for _ in range(n_mixed):
        a, b = rng.choice(4, 2, replace=False)
        pa = rng.choice(12, 2, replace=False)
        pb = rng.choice(12, 2, replace=False)
        parts = [SENTENCES[a][j] for j in pa] + [SENTENCES[b][j] for j in pb]
        rng.shuffle(parts)
        docs.append(" . ".join(parts))
        lab.append(-1)
    order = rng.permutation(len(docs))
    return [docs[j] for j in order], np.array(lab)[order]

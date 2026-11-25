"""Numbers quoted in the text of Chapter 8 that no listing prints."""
import math, numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# Example 8.4-1: distances from z = (5, 5) to the two class means
zA, zB = (5 - 3) ** 2 + (5 - 7) ** 2, (5 - 7) ** 2 + (5 - 4) ** 2
print('8.4-1 distance to A', round(math.sqrt(zA), 2), 'to B', round(math.sqrt(zB), 2))
print('8.4-1 squared distance in variance units A', round(zA / (2 / 3), 1), 'B', round(zB / (1 / 2), 1))
# var_smoothing of GaussianNB on the wine data
X, y = load_wine(return_X_y=True)
g = GaussianNB().fit(X, y)
own = np.array([X[y == c].var(axis=0) for c in g.classes_])
print('var_smoothing default', g.var_smoothing, 'epsilon_', float('%.2g' % g.epsilon_))
print('smallest variance', float('%.2g' % own.min()), 'largest feature variance', float('%.2g' % X.var(axis=0).max()))
print('var_ equals Eq. 8.2-6 plus epsilon_', bool(np.allclose(g.var_, own + g.epsilon_, rtol=0, atol=1e-12)))
# k-NN on the digits split of Listing 8.6
Xd, yd = load_digits(return_X_y=True)
a, b, c, d = train_test_split(Xd, yd, test_size=0.25, random_state=0, stratify=yd)
print('digits k-NN accuracy', round(KNeighborsClassifier().fit(a, c).score(b, d), 3))

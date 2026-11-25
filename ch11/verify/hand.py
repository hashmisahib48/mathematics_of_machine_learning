# Numbers of the slates and of Example 11.4-3, computed.
import numpy as np
y = np.array([3, 5, 4, 8, 7, 9.0])
folds = [[0, 3], [1, 4], [2, 5]]
a = []
for f in folds:
    tr = [i for i in range(6) if i not in f]
    m = y[tr].mean(); e = np.abs(y[f] - m)
    print('fold', [i + 1 for i in f], 'training mean', m, 'errors', e, 'MAE', e.mean())
    a.append(e.mean())
print('a', a, 'CV', np.mean(a))
P, R = 1.0, 0.1
print('slate: arithmetic mean', (P + R) / 2, 'F1', round(2 * P * R / (P + R), 2))

# The update of toy.py, reproduced by scikit-learn (not a listing).
import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier

warnings.simplefilter("ignore")
X8 = np.array([[-2, -1], [-1, -2], [1, 2], [2, 1],
               [-2, 1], [-1, 2], [1, -2], [2, -1]], dtype=float)
y8 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
net = MLPClassifier((2,), solver="sgd", momentum=0, alpha=0,
                    learning_rate_init=0.5, batch_size=1, max_iter=1,
                    random_state=0).fit(X8, y8)
net.coefs_[0][:] = [[0.5, 1.0], [-1.0, 0.5]]   # the weights of toy.py
net.intercepts_[0][:] = [0.5, -0.5]
net.coefs_[1][:] = [[1.0], [-1.0]]
net.intercepts_[1][:] = [0.0]
x = np.array([[2.0, -1.0]])
print("P(class 1) before:", net.predict_proba(x)[0, 1].round(4))
net.partial_fit(x, [1])                         # one step on one sample
print("W1", net.coefs_[0].round(4).tolist(), "b1", net.intercepts_[0].round(4))
print("W2", net.coefs_[1].ravel().round(4), "b2", net.intercepts_[1].round(4))
print("P(class 1) after: ", net.predict_proba(x)[0, 1].round(4))

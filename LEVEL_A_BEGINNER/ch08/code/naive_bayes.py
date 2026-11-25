# Gaussian and multinomial naive Bayes in plain Python.
import math


class GaussianNB:
    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n = len(y)
        self.prior_, self.theta_, self.var_ = {}, {}, {}
        for c in self.classes_:
            R = [x for x, t in zip(X, y) if t == c]
            self.prior_[c] = len(R) / n                        # Step 1
            m = [sum(col) / len(R) for col in zip(*R)]          # Step 2
            v = [sum((a - mu) ** 2 for a in col) / len(R)
                 for col, mu in zip(zip(*R), m)]
            self.theta_[c], self.var_[c] = m, v
        return self

    def _joint(self, z, c):                                     # Step 3
        s = math.log(self.prior_[c])
        for x, m, v in zip(z, self.theta_[c], self.var_[c]):
            s -= 0.5 * math.log(2 * math.pi * v)
            s -= (x - m) ** 2 / (2 * v)
        return s

    def predict_proba(self, Z):
        out = []
        for z in Z:
            L = [self._joint(z, c) for c in self.classes_]
            mx = max(L); e = [math.exp(a - mx) for a in L]      # Step 4
            out.append([a / sum(e) for a in e])
        return out

    def predict(self, Z):                                       # Step 5
        return [self.classes_[max(range(len(p)), key=p.__getitem__)]
                for p in self.predict_proba(Z)]


class MultinomialNB(GaussianNB):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = sorted(set(y)); n, p = len(y), len(X[0])
        self.prior_, self.logp_ = {}, {}
        for c in self.classes_:
            R = [x for x, t in zip(X, y) if t == c]
            self.prior_[c] = len(R) / n
            N = [sum(col) for col in zip(*R)]                   # counts
            tot = sum(N) + self.alpha * p
            self.logp_[c] = [math.log((k + self.alpha) / tot)
                             for k in N]
        return self

    def _joint(self, z, c):
        return math.log(self.prior_[c]) + sum(
            k * lp for k, lp in zip(z, self.logp_[c]))

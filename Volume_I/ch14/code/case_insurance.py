# Case study: claim frequency of a simulated motor portfolio.
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
from portfolio import make_portfolio, TRUE

X, expo, count, amount = make_portfolio()
print(f"policies {len(X)}, claims {count.sum()}, "
      f"exposure {expo.sum():.0f} years, "
      f"policies with a claim {np.mean(count > 0):.1%}")
freq = count / expo                         # claims per year
tr, te = train_test_split(np.arange(len(X)), test_size=0.25,
                          random_state=0)
Xtr, Xte, wtr, wte = X.iloc[tr], X.iloc[te], expo[tr], expo[te]


def report(name, mu):                       # weighted test scores
    dev = mean_poisson_deviance(freq[te], mu, sample_weight=wte)
    d2 = d2_tweedie_score(freq[te], mu, sample_weight=wte, power=1)
    print(f"{name:14s} deviance {dev:.4f}   D2 {d2:.4f}")


const = DummyRegressor().fit(Xtr, freq[tr], sample_weight=wtr)
report("constant", const.predict(Xte))
ols = LinearRegression().fit(Xtr, freq[tr], sample_weight=wtr)
mu = ols.predict(Xte)
print("OLS predictions <= 0 on the test set:", int(np.sum(mu <= 0)))
report("OLS, clipped", np.clip(mu, 1e-3, None))
glm = PoissonRegressor(alpha=1e-4, tol=1e-8, max_iter=300)
glm.fit(Xtr, freq[tr], sample_weight=wtr)   # y = claims/exposure
report("Poisson GLM", glm.predict(Xte))

claims_fit = np.sum(wtr * glm.predict(Xtr))  # balance, Eq. 14.3-10
print(f"training claims: observed {count[tr].sum()}, "
      f"fitted {claims_fit:.1f}")
print(f"base frequency exp(b) = {np.exp(glm.intercept_):.3f}")
for name, w, t in zip(X.columns, glm.coef_, TRUE.values()):
    print(f"{name:11s} exp(w) = {np.exp(w):.3f}   "
          f"true {np.exp(t):.3f}")

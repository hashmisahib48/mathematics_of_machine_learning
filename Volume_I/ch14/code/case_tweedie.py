# Case study, part 2: the pure premium with Tweedie regressors.
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import train_test_split
from portfolio import make_portfolio

X, expo, count, amount = make_portfolio()
prem = amount / expo                        # cost per year insured
tr, te = train_test_split(np.arange(len(X)), test_size=0.25,
                          random_state=0)
print(f"policies with zero cost: {np.mean(amount == 0):.1%}")
print("power   D2 at 1.5   observed cost   predicted cost")
for power in (1.1, 1.5, 1.9):
    m = TweedieRegressor(power=power, alpha=1e-4, tol=1e-8,
                         max_iter=300)
    m.fit(X.iloc[tr], prem[tr], sample_weight=expo[tr])
    mu = m.predict(X.iloc[te])
    d2 = d2_tweedie_score(prem[te], mu, sample_weight=expo[te],
                          power=1.5)
    print(f"{power:5.1f}   {d2:9.4f}   {np.sum(amount[te]):13.0f}"
          f"   {np.sum(expo[te] * mu):14.0f}")

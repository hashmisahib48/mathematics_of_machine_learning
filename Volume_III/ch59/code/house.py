# The invented house table of Chapter 59 (made in code, not real).
import numpy as np
import pandas as pd

KINDS = ["flat", "terrace", "house"]
CONDS = ["poor", "fair", "good", "new"]


def make_houses(n=3000, seed=0, n_districts=40):
    """Columns: area (m2), rooms, age (years), dist (km to centre),
    kind, condition, garden, district, agent_fee, price (thousands)."""
    rng = np.random.RandomState(seed)
    dnames = np.array([f"D{j:02d}" for j in range(1, n_districts + 1)])
    d_eff = np.random.RandomState(7).normal(0, 0.22, n_districts)
    w = 1 / np.arange(1, n_districts + 1) ** 0.7
    di = rng.choice(n_districts, n, p=w / w.sum())
    kind = rng.choice(KINDS, n, p=[.45, .30, .25])
    cond = rng.choice(CONDS, n, p=[.10, .30, .40, .20])
    k_eff = dict(flat=-0.10, terrace=0.0, house=0.12)
    c_eff = dict(poor=-0.20, fair=-0.07, good=0.04, new=0.15)
    area = np.exp(rng.normal(4.3, 0.35, n)) * np.where(
        kind == "house", 1.4, np.where(kind == "terrace", 1.1, 0.8))
    rooms = np.clip(np.round(area / 28 + rng.normal(0, .6, n)), 1, 9)
    age = np.round(np.minimum(rng.gamma(2.0, 18, n), 120))
    dist = np.round(rng.gamma(2.5, 3.0, n), 1)
    garden = np.where(kind == "flat", rng.rand(n) < .05,
                      rng.rand(n) < .7)
    logp = (1.7 + 0.9 * np.log(area) - 0.004 * age - 0.02 * dist
            + 0.05 * garden + d_eff[di]
            + np.array([k_eff[k] for k in kind])
            + np.array([c_eff[c] for c in cond])
            + rng.normal(0, 0.15, n))
    price = np.round(np.exp(logp), 1)
    df = pd.DataFrame(dict(
        area=np.round(area, 1), rooms=rooms, age=age, dist=dist,
        kind=kind, condition=cond, garden=np.where(garden, "yes", "no"),
        district=dnames[di], agent_fee=np.round(0.02 * price, 2),
        price=price))
    # holes: area MCAR, age MAR (more often missing for terraces and
    # houses), condition missing when poor (MNAR) or at random
    df.loc[rng.rand(n) < .04, "area"] = np.nan
    df.loc[rng.rand(n) < np.where(kind == "flat", .02, .12), "age"] = np.nan
    df.loc[(rng.rand(n) < .06) | ((cond == "poor") & (rng.rand(n) < .25)),
           "condition"] = np.nan
    # 30 listings entered twice
    return pd.concat([df, df.iloc[rng.choice(n, 30, replace=False)]],
                     ignore_index=True)

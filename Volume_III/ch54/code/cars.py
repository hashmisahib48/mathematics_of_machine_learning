# The invented table of used cars (Chapter 50). make_cars(n, seed) is
# copied from Chapter 50 for Chapter 54. The population is fixed (city effects use their
# own seed); `seed` only chooses which n cars are drawn.
import os
import numpy as np
import pandas as pd

COLOURS = ["white", "black", "silver", "blue", "red", "green"]
FUELS = ["petrol", "diesel", "hybrid", "electric"]
TRIMS = ["basic", "comfort", "sport", "luxury"]
FUEL_PREMIUM = dict(petrol=0.0, diesel=0.8, hybrid=2.5, electric=5.0)
TRIM_FACTOR = dict(basic=1.0, comfort=1.12, sport=1.25, luxury=1.5)

def make_cars(n=2000, seed=0, n_cities=200):
    """Columns: colour, fuel, trim, city, age (years), km (thousand
    km), price (thousand currency units)."""
    rng = np.random.RandomState(seed)
    cities = np.array([f"C{j:04d}" for j in range(1, n_cities + 1)])
    city_effect = np.random.RandomState(2024).normal(0, 3.0, n_cities)
    w = 1 / np.arange(1, n_cities + 1) ** 0.8       # a few big cities
    ci = rng.choice(n_cities, n, p=w / w.sum())
    colour = rng.choice(COLOURS, n, p=[.28, .24, .20, .12, .10, .06])
    fuel = rng.choice(FUELS, n, p=[.45, .30, .17, .08])
    trim = rng.choice(TRIMS, n, p=[.35, .35, .20, .10])
    age = np.round(np.minimum(rng.gamma(2.2, 2.2, n), 20), 1)
    km = np.round(age * 12 * rng.lognormal(0, 0.35, n), 1)
    base = 32 * np.exp(-0.14 * age) * np.array(
        [TRIM_FACTOR[t] for t in trim])
    price = (base + np.array([FUEL_PREMIUM[f] for f in fuel])
             + city_effect[ci] - 0.02 * km + rng.normal(0, 1.2, n))
    return pd.DataFrame(dict(
        colour=colour, fuel=fuel, trim=trim, city=cities[ci],
        age=age, km=km, price=np.round(np.maximum(price, 1.0), 2)))

if __name__ == "__main__":
    df = make_cars(2000, 0)
    here = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(here, "..", "verify", "cars_2000_seed0.csv"),
              index=False)
    print(df.head(8).to_string())
    print(df.describe().round(2).to_string())
    print(df.city.value_counts().iloc[[0, 1, 4, 9, 19, 39]].to_string())

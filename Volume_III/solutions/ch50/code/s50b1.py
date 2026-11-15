# Problem 50B.1: grouping rare cities with min_frequency
import sys
sys.path.insert(0, "../../../ch50/code")   # make_cars
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from cars import make_cars

train, _ = train_test_split(make_cars(4000, 0), random_state=0)
city = train[["city"]]
counts = city["city"].value_counts()          # cars per city
print("training cars:", len(city), "  cities:", len(counts))
print("cars per city: min", counts.min(), " max", counts.max())
print(f"\n{'k':>3}{'columns':>9}{'grouped':>9}{'cars':>7}"
      f"{'share':>8}{'by hand':>10}")
for k in (1, 5, 10, 20, 50):
    enc = OneHotEncoder(min_frequency=k, sparse_output=False)
    cols = enc.fit(city).transform(city).shape[1]
    rare = enc.infrequent_categories_[0]   # None: no group
    n_rare = 0 if rare is None else len(rare)
    cars = 0 if rare is None else counts[rare].sum()
    by_hand = int((counts < k).sum())      # cities < k cars
    hand_cols = len(counts) - by_hand + (by_hand > 0)
    share = cars / len(city)
    print(f"{k:3d}{cols:9d}{n_rare:9d}{cars:7d}{share:8.3f}"
          f"{by_hand:7d}/{hand_cols}")

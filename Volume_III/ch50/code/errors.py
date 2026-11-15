# The messages quoted in the COMMON ERROR box and in the text.
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from toy import cars

def show(tag, fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            print(tag, type(w[0].message).__name__ + ":")
            print("   ", str(w[0].message)[:62], "...")
        except Exception as e:
            print(tag, type(e).__name__ + ":")
            print("   ", str(e)[:62], "...")

new = pd.DataFrame({"colour": ["green"], "fuel": ["petrol"]})
oh = OneHotEncoder().fit(cars[["colour", "fuel"]])
show("unseen colour", lambda: oh.transform(new))

te = TargetEncoder(random_state=0, target_type="continuous")
show("random_state", lambda: te.fit_transform(cars[["colour"]],
                                              cars.price))

big = pd.DataFrame({"city": pd.Categorical(np.arange(300) % 300),
                    "x": np.arange(300.)})
hgb = HistGradientBoostingRegressor(categorical_features="from_dtype")
show("300 categories", lambda: hgb.fit(big, np.arange(300.)))

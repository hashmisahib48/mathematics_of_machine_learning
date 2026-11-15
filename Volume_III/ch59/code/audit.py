# Section 59.3: the data audit, before any model is fitted.
import numpy as np
import pandas as pd
from house import make_houses

df = make_houses()
y = df["price"]
print(f"rows {len(df)}, columns {df.shape[1]}, "
      f"duplicate rows {df.duplicated().sum()}")
num = df.select_dtypes("number").drop(columns="price")
rho = num.apply(lambda s: abs(s.corr(y, method="spearman")))
print(pd.DataFrame({"dtype": df.dtypes.astype(str),
                    "levels": df.nunique(),
                    "missing%": (100 * df.isna().mean()).round(1),
                    "|rho|": rho.round(2)}).to_string())
print(f"price: median {y.median():.0f}, mean {y.mean():.0f}, "
      f"skew {y.skew():.2f}, log skew {np.log(y).skew():.2f}")

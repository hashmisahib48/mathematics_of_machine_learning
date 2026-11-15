from sklearn.model_selection import KFold
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   TargetEncoder, KBinsDiscretizer)
from toy import cars

cat, y = cars[["colour", "fuel"]], cars.price
col, age = cars[["colour"]], cars[["age"]]
new = cat.iloc[[0]].assign(colour="green")       # a colour not seen
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
print(oh.fit(cat).get_feature_names_out().tolist())
print("green ->", oh.transform(new)[0])
d1 = OneHotEncoder(drop="first", sparse_output=False).fit(cat)
print(d1.get_feature_names_out().tolist())
od = OrdinalEncoder(handle_unknown="use_encoded_value",
                    unknown_value=-1).fit(cat)
print("codes", od.transform(cat)[:2].tolist(),
      "green", od.transform(new))

kf = KFold(4, shuffle=True, random_state=0)
te = TargetEncoder(smooth=2.0, target_type="continuous", cv=kf)
Z = te.fit_transform(col, y)[:, 0]
print("encodings_", te.encodings_[0], " mean", te.target_mean_)
print("fit.transform ", te.transform(col)[:, 0].round(2))
print("fit_transform ", Z.round(2))

kq = KBinsDiscretizer(4, encode="ordinal", strategy="quantile")
print("bins", kq.fit_transform(age)[:, 0], kq.bin_edges_[0])

# Problem 59B.1: one-hot or target encoding for the 40 districts
import sys
sys.path.insert(0, "../../../ch59/code")      # the chapter's helpers
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from case_b1 import Xtr, ytr, hgb, log        # data, model

def folds(model):                             # MAE of each of 5 folds
    return -cross_val_score(model, Xtr, ytr, cv=5,
                            scoring="neg_mean_absolute_error")

target = folds(log(hgb()))                    # book: TargetEncoder
onehot_model = log(hgb())                     # one change
onehot_model.regressor.steps[0][1].set_params(
    district=OneHotEncoder(handle_unknown="ignore",
                           sparse_output=False))
onehot = folds(onehot_model)

counts = Xtr["district"].value_counts()
print("districts", len(counts), " rows", len(Xtr),
      " rows per district: min", counts.min(),
      " mean %.1f" % counts.mean(), " max", counts.max())
print("target encoding  1 column : MAE %.2f" % target.mean())
print("one-hot         40 columns: MAE %.2f" % onehot.mean())
print("folds, target :", target.round(2))
print("folds, one-hot:", onehot.round(2))
d = onehot - target
print("difference per fold:", d.round(2), " mean %.2f" % d.mean())
print("sd of the target-encoding folds %.2f" % target.std(ddof=1))

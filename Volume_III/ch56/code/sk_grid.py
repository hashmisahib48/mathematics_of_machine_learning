# Section 56.7: GridSearchCV with two scores and train scores.
import numpy as np
from sklearn.model_selection import GridSearchCV
from common import cancer_split, svc_pipe, GRID

Xtr, Xte, ytr, yte = cancer_split()
gs = GridSearchCV(svc_pipe(), GRID, cv=5, refit="acc",
                  scoring={"acc": "accuracy", "f1": "f1"},
                  return_train_score=True).fit(Xtr, ytr)
r = gs.cv_results_
print(len(r), "keys; for 'acc':",
      [k for k in r if k.endswith("_acc") and "split" not in k])
print("multimetric_", gs.multimetric_, " best_index_", gs.best_index_)
print("best_score_", round(gs.best_score_, 4), " test",
      round(gs.score(Xte, yte), 4), "(accuracy, the refit scorer)")
print("rank  log2 C  log2 g   acc    sd   train     f1")
for i in np.argsort(r["rank_test_acc"], kind="stable")[:4]:
    p = r["params"][i]
    row = [r[k][i] for k in ("mean_test_acc", "std_test_acc",
                             "mean_train_acc", "mean_test_f1")]
    print(f"{r['rank_test_acc'][i]:4d}{np.log2(p['svc__C']):8.0f}"
          f"{np.log2(p['svc__gamma']):8.0f}",
          *(f"{v:7.4f}" for v in row))

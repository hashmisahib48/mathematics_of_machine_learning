# Problem 55C.1: why balanced (stratified) folds do not stop the leak between visits of one patient
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/../../../ch55/code')
from cvdata import make_patients
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score, cross_val_predict
rf = RandomForestClassifier(50, random_state=0)

st, gr, new, shares, balance = [], [], [], [], []
for seed in range(8):
    X, y, g = make_patients(40, 8, seed=seed); Xn, yn, _ = make_patients(150, 8, seed=100 + seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    st.append(cross_val_score(rf, X, y, cv=cv).mean())
    gr.append(cross_val_score(rf, X, y, groups=g, cv=GroupKFold(5)).mean())
    new.append(rf.fit(X, y).score(Xn, yn))
    shares.append(np.mean([np.isin(g[te], g[tr]).mean() for tr, te in cv.split(X, y)]))
    balance.append(max(abs(y[te].mean() - y.mean()) for _, te in cv.split(X, y)))
print("8 invented hospitals, 40 patients x 8 visits")
print("  StratifiedKFold accuracy %.3f   GroupKFold %.3f   new patients %.3f" % (np.mean(st), np.mean(gr), np.mean(new)))
print("  largest deviation of a fold's class-1 share from the whole: %.4f (mean over hospitals)" % np.mean(balance))
print("  share of test rows whose patient is also in training (stratified): %.3f" % np.mean(shares))

# can the forest tell the patients apart from the six numbers alone?  (class is not used)
X, y, g = make_patients(40, 8, seed=0)
pid = cross_val_predict(RandomForestClassifier(50, random_state=0), X, g, cv=StratifiedKFold(4, shuffle=True, random_state=0))
print("\nhospital 0: patient identified from one visit, 4-fold, 40 candidates: %.3f   (chance %.3f)" % ((pid == g).mean(), 1 / 40))
# how much of each feature's variance lies between patients
m = np.array([X[g == i].mean(0) for i in range(40)])
print("share of variance between patients, per feature:", np.round(m.var(0, ddof=1) / X.var(0, ddof=1), 2).tolist())

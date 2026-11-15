# Problem 34B.3: thresholds of self-training, ten labels.
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier

C = [[-1.5, 0], [1.5, 0]]
X, y = make_blobs(400, centers=C, cluster_std=1.6, random_state=5)
Xt, yt = make_blobs(4000, centers=C, cluster_std=1.6,
                    random_state=777)
rng = np.random.default_rng(5)
idx = np.r_[rng.choice(np.where(y == 0)[0], 5, replace=False),
            rng.choice(np.where(y == 1)[0], 5, replace=False)]
yy = -np.ones(400, int)
yy[idx] = y[idx]                     # five labels per class

print(f"best possible, x1 = 0: {np.mean((Xt[:, 0] > 0) == yt):.4f}")
base = LogisticRegression().fit(X[idx], y[idx])
print(f"ten labels alone:      {base.score(Xt, yt):.4f}")
print("  t  rounds  pseudo  left  stop       right  test acc")
for t in [0.6, 0.8, 0.9, 0.95, 0.99]:
    st = SelfTrainingClassifier(LogisticRegression(),
                                threshold=t, max_iter=None)
    st.fit(X, yy)
    ps = st.labeled_iter_ > 0        # labelled by the loop
    ok = np.mean(st.transduction_[ps] == y[ps])
    print(f"{t:4.2f} {st.n_iter_:5d} {ps.sum():7d} "
          f"{(st.transduction_ == -1).sum():5d}  "
          f"{st.termination_condition_:9s} {ok:6.3f} "
          f"{st.score(Xt, yt):9.4f}")

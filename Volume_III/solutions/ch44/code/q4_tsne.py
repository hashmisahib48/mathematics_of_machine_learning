# Numbers for Question 4 of Chapter 44: sizes and gaps in a t-SNE map
import numpy as np
from sklearn.manifold import TSNE
np.set_printoptions(precision=4, suppress=True)

print("== Q4  t-SNE: cluster sizes and gaps")
rngb = np.random.RandomState(1)
sd = [0.5, 3.0, 0.5]; mu = [np.r_[0, 0, 0, 0, 0], np.r_[8, 0, 0, 0, 0], np.r_[40, 0, 0, 0, 0]]
B = np.vstack([rngb.standard_normal((150, 5)) * s + m for s, m in zip(sd, mu)]); yb = np.repeat([0, 1, 2], 150)
print("true sd per coordinate:", sd, " true centres on the first axis: 0, 8, 40")
for per in (10, 30, 100):  # three runs
    Z = TSNE(2, perplexity=per, init="pca", random_state=0).fit_transform(B)
    c = np.array([Z[yb == k].mean(0) for k in range(3)]); rad = np.array([np.sqrt(((Z[yb == k] - c[k]) ** 2).sum(1).mean()) for k in range(3)])
    print(f"perplexity {per:3d}: map radius per cluster {rad.round(1)}   distances between centres 1-2 {np.linalg.norm(c[0]-c[1]):5.1f}  2-3 {np.linalg.norm(c[1]-c[2]):5.1f}  1-3 {np.linalg.norm(c[0]-c[2]):5.1f}  ratio of the gaps {np.linalg.norm(c[1]-c[2])/np.linalg.norm(c[0]-c[1]):.2f}")
print("true radius per cluster (sqrt(5) sd):", (np.sqrt(5) * np.array(sd)).round(2), " true centre distances 8, 32, 40")

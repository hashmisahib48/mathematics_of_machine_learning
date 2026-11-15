# Three doctors, five patients: hard and soft voting by hand.
import numpy as np

# rows: patients; columns: doctors A, B, C; entry: P(disease)
P = np.array([[0.90, 0.80, 0.70],
              [0.55, 0.35, 0.40],
              [0.95, 0.30, 0.40],
              [0.20, 0.10, 0.30],
              [0.55, 0.60, 0.10]])
y = np.array([1, 0, 1, 0, 1])               # what the biopsy showed
w = np.array([2.0, 1.0, 1.0])               # A counts twice

votes = (P > 0.5).astype(int)               # each doctor's own call
hard = (votes.sum(1) > 1.5).astype(int)     # Eq. 29.3-1, 3 voters
soft = P.mean(1)                            # Eq. 29.3-2
soft_w = P @ w / w.sum()                    # weighted average
print("votes   ", votes.tolist())
for name, q in (("hard", hard), ("soft", soft > 0.5),
                ("soft w", soft_w > 0.5)):
    print(f"{name:7s}", q.astype(int), " correct", (q == y).sum())
print("soft    ", soft.round(4))
print("soft w  ", soft_w.round(4))
for j, d in enumerate("ABC"):
    print("doctor", d, "correct", (votes[:, j] == y).sum())

def log_loss(q):
    return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

each = [log_loss(P[:, j]) for j in range(3)]
print("log loss A, B, C:", np.round(each, 4),
      " mean", round(np.mean(each), 4))
print("log loss of the soft vote:", round(log_loss(soft), 4))

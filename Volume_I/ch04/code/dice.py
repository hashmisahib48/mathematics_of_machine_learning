# Counting equally likely outcomes of two dice (§4.1, §4.2).
import itertools

S = list(itertools.product(range(1, 7), repeat=2))   # 36 outcomes
A = {s for s in S if sum(s) == 7}                    # the sum is 7
B = {s for s in S if 6 in s}                         # a six shows
n = len(S)
print("|S|, |A|, |B|, |A and B| =", n, len(A), len(B), len(A & B))
print(f"P(A) = {len(A) / n:.4f}   P(B) = {len(B) / n:.4f}")
print(f"P(no six) = {25 / n:.4f}   1 - P(B) = {1 - len(B) / n:.4f}")
print(f"P(A or B) = {len(A | B) / n:.4f}")       # Eq. 4.1-2
print(f"P(A and B) = {len(A & B) / n:.4f}")
print(f"P(A|B) = {len(A & B) / len(B):.4f}")     # Eq. 4.2-1

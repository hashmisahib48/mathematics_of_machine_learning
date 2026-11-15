# Decoding an error-correcting output code by hand (Section 33.5).
import numpy as np

M = np.array([[ 1,  1,  1,  1,  1,  1],      # class A
              [ 1,  1, -1, -1, -1, -1],      # class B
              [-1, -1,  1,  1, -1, -1],      # class C
              [-1, -1, -1, -1,  1,  1]])     # class D
ham = (M[:, None, :] != M[None, :, :]).sum(2)
print("Hamming distances between code words:\n", ham)

s = np.array([0.8, 0.9, -0.7, -0.6, 0.4, -0.9])   # six scores
print("signs of the scores:", np.sign(s).astype(int))
d = np.sqrt(((s - M) ** 2).sum(1))                # Eq. 33.3-10
print("Euclidean distances :", d.round(3))
print("nearest code word   :", "ABCD"[d.argmin()])
print("scores that agree in sign with code B:",
      int((np.sign(s) == M[1]).sum()), "of 6")

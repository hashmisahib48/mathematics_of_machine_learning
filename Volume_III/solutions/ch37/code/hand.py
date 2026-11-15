# Hand checks for Problems 37A.1 - 37A.4 and 37C.2 (numbers quoted in the solutions).
import numpy as np
from scipy.cluster.hierarchy import linkage

X7 = np.array([[3,3],[1,2],[0,2],[7,4],[11,2],[5,12],[8,9]], float)
X = X7[:5]; nm = "ABCDE"
d = lambda a, b: float(np.linalg.norm(a - b))
print("== 37A.1  average linkage, species A to E")
D = {(i, j): d(X[i], X[j]) for i in range(5) for j in range(i+1, 5)}
for (i, j), v in D.items(): print(f"  d({nm[i]},{nm[j]}) = {v:.4f}")
dA_BC = (D[(0,1)] + D[(0,2)]) / 2; dD_BC = (D[(1,3)] + D[(2,3)]) / 2; dE_BC = (D[(1,4)] + D[(2,4)]) / 2
print(f"  after BC:  d(A,BC) = (2.236+3.162)/2 = {dA_BC:.4f}   d(D,BC) = {dD_BC:.4f}   d(E,BC) = {dE_BC:.4f}")
dD_ABC = (1*D[(0,3)] + 2*dD_BC) / 3; dE_ABC = (1*D[(0,4)] + 2*dE_BC) / 3
print(f"  after ABC: d(D,ABC) = (4.123+2*{dD_BC:.4f})/3 = {dD_ABC:.4f}   d(E,ABC) = {dE_ABC:.4f}   d(D,E) = {D[(3,4)]:.4f}")
last = (dD_ABC + dE_ABC) / 2
direct = np.mean([d(X[i], X[j]) for i in (0,1,2) for j in (3,4)])
print(f"  last merge  d(ABC,DE) = ({dD_ABC:.4f}+{dE_ABC:.4f})/2 = {last:.4f};  mean of the six distances = {direct:.4f}")
print("  scipy heights:", np.round(linkage(X, 'average')[:, 2], 4).tolist())
print("  sum of the six distances:", round(sum(d(X[i], X[j]) for i in (0,1,2) for j in (3,4)), 4))

print("\n== 37A.2  Ward for P, Q, R, S")
P = np.array([[0,0],[2,0],[0,4],[6,4]], float); pn = "PQRS"
for i in range(4):
    for j in range(i+1, 4):
        print(f"  Delta({pn[i]},{pn[j]}) = 1/2 * {((P[i]-P[j])**2).sum():.0f} = {0.5*((P[i]-P[j])**2).sum():.4f}")
mu = P[:2].mean(0); print("  centroid of PQ:", mu)
for k in (2, 3):
    dd = ((P[k]-mu)**2).sum(); print(f"  |{pn[k]} - mu_PQ|^2 = {dd:.4f};  Delta(PQ,{pn[k]}) = 2*1/3*{dd:.4f} = {2/3*dd:.4f}  height {np.sqrt(2*2/3*dd):.4f}")
mu3 = P[:3].mean(0); dd = ((P[3]-mu3)**2).sum(); print(f"  centroid of PQR {mu3.round(4)}; |S-mu|^2 = {dd:.4f}; Delta = 3*1/4*{dd:.4f} = {0.75*dd:.4f}; height {np.sqrt(1.5*dd):.4f}")
print("  Lance-Williams check of Delta(PQ,R): squared dist 17=|R-mu|^2? ->  (2/3)*16 + (2/3)*20... see below")
d2 = lambda a, b: ((P[a]-P[b])**2).sum()
lw = (2/3)*d2(0,2) + (2/3)*d2(1,2) - (1/3)*d2(0,1)     # (n_A+n_C)/T d2(A,C)+(n_B+n_C)/T d2(B,C) - n_C/T d2(A,B)
print(f"  L-W for d^2(PQ,R): (2/3)*{d2(0,2):.0f} + (2/3)*{d2(1,2):.0f} - (1/3)*{d2(0,1):.0f} = {lw:.4f} = 2*Delta = {2*2/3*((P[2]-mu)**2).sum():.4f}")
SS = ((P - P.mean(0))**2).sum(); print("  overall centroid", P.mean(0), " SS_total =", SS, " per point:", ((P-P.mean(0))**2).sum(1).tolist())
print("  sum of costs:", round(2 + 2/3*17 + 0.75*320/9, 4))
print("  scipy ward heights:", np.round(linkage(P, 'ward')[:, 2], 4).tolist())
# what if R and S were merged first (cost 18) -- the greedy choice is not the best total only in later steps
print("  alternatives at step 2: R+S", 18.0, " PQ+R", round(2/3*17,4), " PQ+S", round(2/3*((P[3]-mu)**2).sum(),4))

print("\n== 37A.3  clustering features")
A = np.array([[1,1],[1,3]], float); B = np.array([[3,1],[3,3]], float)
cf = lambda M: (len(M), M.sum(0), (M**2).sum())
c1, c2 = cf(A), cf(B); print("  CF1", c1[0], c1[1], c1[2], " CF2", c2[0], c2[1], c2[2])
cm = (c1[0]+c2[0], c1[1]+c2[1], c1[2]+c2[2]); print("  sum", cm[0], cm[1], cm[2], " centroid", cm[1]/cm[0])
for nme, c in (("CF1", c1), ("CF2", c2), ("sum", cm)):
    cen = c[1]/c[0]; R2 = c[2]/c[0] - cen@cen; print(f"  {nme}: SS/N = {c[2]/c[0]:.4f}  |centroid|^2 = {cen@cen:.4f}  R^2 = {R2:.4f}  R = {np.sqrt(R2):.4f}")
for T in (1.5, 1.2): print(f"  threshold {T}: R^2 = 2 <= {T**2:.2f}? {2 <= T**2}")
direct = np.sqrt(((np.r_[A, B] - np.r_[A, B].mean(0))**2).sum(1).mean()); print("  direct radius of the four points:", round(direct, 4))

print("\n== 37A.4  cutting a chain")
x = np.array([0, .8, 2.1, 4.1, 6.2, 11.6]); gaps = np.diff(x); print("  gaps", np.round(gaps, 2).tolist())
Z = linkage(x[:, None], 'single'); h = Z[:, 2]; print("  single-linkage heights", np.round(h, 2).tolist())
for t in (1.5, 2.1, 3.0, 6.0): print(f"  t={t}: heights >= t: {np.round(h[h>=t],2).tolist()}  K = 1 + {int((h>=t).sum())} = {1+int((h>=t).sum())}")

print("\n== 37B.2  the inversion by hand")
Q = np.array([[0, 0], [2, 0], [1, 1.7]]); sq = lambda a, b: float(((Q[a]-Q[b])**2).sum())
print("  squared distances: 1-2 %.4f  1-3 %.4f  2-3 %.4f" % (sq(0,1), sq(0,2), sq(1,2)))
new = 0.5*sq(0,1) + 0.5*sq(1,2) - 0.25*sq(0,2)
print("  merge points 1 and 3 (the tie goes to the first pair): height %.4f;  d^2(13,2) = 1/2*%.4f + 1/2*%.4f - 1/4*%.4f = %.4f  height %.4f" % (np.sqrt(sq(0,2)), sq(0,1), sq(1,2), sq(0,2), new, np.sqrt(new)))
c13 = (Q[0]+Q[2])/2; print("  centroid of 1 and 3:", c13, " its distance to point 2: %.4f" % np.linalg.norm(c13-Q[1]))

print("\n== slips")
wD = (D[(0,3)] + dD_BC) / 2; wE = (D[(0,4)] + dE_BC) / 2
print("  37A.1 unweighted update at step 3: d(D,ABC) = %.4f  d(E,ABC) = %.4f  last height %.4f (correct 7.7983)" % (wD, wE, (wD + wE) / 2))
print("  37A.4 floating point gaps:", [repr(float(g)) for g in np.diff(x)])
print("  37A.2 heights taken as Delta instead of sqrt(2 Delta): 2, 11.3333, 26.6667")

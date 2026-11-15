# Vectors, norms, dot product and angle (Sections 2.1 - 2.3).
import numpy as np

u = np.array([3.0, 1.0])
v = np.array([1.0, 2.0])
print("u + v =", u + v, "  2u - v =", 2 * u - v)    # Eq. 2.1-1

w = np.array([3.0, -4.0, 12.0])
for q in (2, 1, np.inf):                              # Eq. 2.2-1
    print(f"norm {q}: {np.linalg.norm(w, q):.1f}")

dot = u @ v                                           # Eq. 2.3-1
nu, nv = np.linalg.norm(u), np.linalg.norm(v)
cos = dot / (nu * nv)                                 # Eq. 2.3-4
print(f"u.v = {dot:.0f}   |u| = {nu:.4f}   |v| = {nv:.4f}")
ang = np.degrees(np.arccos(cos))
print(f"cos = {cos:.4f}   angle = {ang:.1f} degrees")
print("u . (-1, 3) =", u @ np.array([-1.0, 3.0]))   # square

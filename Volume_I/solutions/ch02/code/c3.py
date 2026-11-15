# Problem 2C.3: distance or direction? three documents over two words
import numpy as np
d = {"d1": np.array([3., 1]), "d2": np.array([6., 2]), "d3": np.array([1., 3])}
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
for a, b in (("d1", "d2"), ("d1", "d3"), ("d2", "d3")):
    diff = d[a] - d[b]
    print(f"{a},{b}: difference {diff}, distance {np.linalg.norm(diff):.3f}, dot {d[a] @ d[b]:.0f}, "
          f"lengths {np.linalg.norm(d[a]):.3f} and {np.linalg.norm(d[b]):.3f}, cosine {cos(d[a], d[b]):.4f}")
print("angle d1,d3 and d2,d3 in degrees:", np.degrees(np.arccos(cos(d["d1"], d["d3"]))).round(2), np.degrees(np.arccos(cos(d["d2"], d["d3"]))).round(2))
print("shares of word 1 (d1, d2, d3):", [round(float(v[0] / v.sum()), 3) for v in d.values()])
print("after dividing each by its length, distance d1,d2:", np.linalg.norm(d["d1"] / np.linalg.norm(d["d1"]) - d["d2"] / np.linalg.norm(d["d2"])))
print("after dividing each by its length, distance d1,d3: %.3f   d2,d3: %.3f" % (
    np.linalg.norm(d["d1"] / np.linalg.norm(d["d1"]) - d["d3"] / np.linalg.norm(d["d3"])),
    np.linalg.norm(d["d2"] / np.linalg.norm(d["d2"]) - d["d3"] / np.linalg.norm(d["d3"]))))

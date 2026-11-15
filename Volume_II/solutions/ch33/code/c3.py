# Problem 33C.3: 5000 classes, counts and the promise of a random code
import numpy as np
K = 5000
print("one-versus-rest problems       :", K)
print("one-versus-one problems        :", K * (K - 1) // 2)
B = int(K * 1.5); print("output code, code_size=1.5 bits:", B)
print("rows in all (n samples): OvR", K, "n; OvO", K - 1, "n; ECOC", B, "n")
print("squared-row cost relative to OvR: OvO", round(2 * (K - 1) / K ** 2, 6), " ECOC", 1.5)
print("predict calls per sample: OvR", K, " OvO", K * (K - 1) // 2, " ECOC", B)

rng = np.random.default_rng(0)
for bits in (7500, 750, 75):
    M = rng.choice(np.array([-1., 1.], dtype=np.float32), size=(K, bits))
    G = M @ M.T                                   # bits - 2 * Hamming distance
    iu = np.triu_indices(K, 1)
    d = ((bits - G[iu]) / 2).astype(np.int32)
    print(f"{bits:5d} bits: mean Hamming distance {d.mean():8.1f}, smallest "
          f"{d.min():5d}, pairs {len(d)}, bits the nearest-word rule "
          f"can always correct {(d.min() - 1) // 2:5d} "
          f"({(d.min() - 1) // 2 / bits:.3f} of the bits)")
    print(f"        binomial prediction: mean {bits / 2:.0f}, sd {np.sqrt(bits) / 2:.1f}")
    del M, G, d

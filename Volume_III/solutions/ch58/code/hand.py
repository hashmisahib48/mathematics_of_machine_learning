# Hand checks for Problems 58A.1 and 58A.2 (every number quoted in the solutions)
import itertools
import numpy as np

print("== 58A.1  a miss costs 19 times a false alarm")
c_fn, c_fp = 19.0, 1.0
t_star = c_fp / (c_fp + c_fn)
print("t* = c_FP/(c_FP + c_FN) = %g/%g = %.4f" % (c_fp, c_fp + c_fn, t_star))
print("log-odds threshold ln(c_FP/c_FN) = ln(1/19) = %.4f   (check: logit(t*) = %.4f)"
      % (np.log(c_fp / c_fn), np.log(t_star / (1 - t_star))))
for p in (0.04, 0.06):
    yes, no = (1 - p) * c_fp, p * c_fn
    print("p = %.2f: say yes costs (1-p)*c_FP = %.2f, say no costs p*c_FN = %.2f -> %s ; p %s t*"
          % (p, yes, no, "yes" if yes < no else "no", ">=" if p >= t_star else "<"))
for p in (0.05, 0.0499, 0.0501):
    print("p = %.4f: yes %.4f  no %.4f" % (p, (1 - p) * c_fp, p * c_fn))
print("at p = t* both actions cost %.4f" % ((1 - t_star) * c_fp))

print("\n== 58A.2  8 rows, 3 positive truths, 5 predicted positive, accuracy 6/8")
y = np.array([1, 1, 1, 0, 0, 0, 0, 0]); yh = np.array([1, 1, 1, 1, 1, 0, 0, 0])
n, n1, n0, m1, m0 = 8, y.sum(), (1 - y).sum(), yh.sum(), (1 - yh).sum()
base = (y == yh).mean()
print("n1 = %d, n0 = %d, m1 = %d, m0 = %d  baseline accuracy = %.4f" % (n1, n0, m1, m0, base))
form = (n1 * m1 + n0 * m0) / n ** 2
print("expected accuracy after a shuffle (Eq. 58.4-2) = (%d*%d + %d*%d)/%d = %d/%d = %.5f"
      % (n1, m1, n0, m0, n ** 2, n1 * m1 + n0 * m0, n ** 2, form))
print("importance I = %.4f - %.5f = %.5f" % (base, form, base - form))
accs = np.array([(yh[list(q)] == y).mean() for q in itertools.permutations(range(8))])
print("average over all 8! = %d orders: %.5f" % (len(accs), accs.mean()))
vals, cnt = np.unique(accs.round(4), return_counts=True)
print("distribution of the accuracy after a shuffle:", dict(zip(vals.tolist(), cnt.tolist())))
print("share of orders with accuracy at least the baseline 0.75: %.4f" % (accs >= 0.75).mean())
# the same through the cell counts: after a shuffle the prediction of row i is a random row's prediction
print("P(prediction agrees with truth) for a positive truth: m1/n = %d/%d; for a negative truth: m0/n = %d/%d"
      % (m1, n, m0, n))
print("sum over rows: %d*(%d/%d) + %d*(%d/%d) = %.3f  ->  divided by n: %.5f"
      % (n1, m1, n, n0, m0, n, n1 * m1 / n + n0 * m0 / n, (n1 * m1 / n + n0 * m0 / n) / n))

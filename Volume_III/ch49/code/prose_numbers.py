# Numbers quoted in the prose of Chapter 49 (not listings).
import json, warnings
import numpy as np
from scipy import sparse
from scipy.special import ndtri
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
    QuantileTransformer, PowerTransformer, FunctionTransformer,
    KernelCenterer, Binarizer)
from sklearn.metrics.pairwise import linear_kernel

N = {}
# 1. distance identity: mean of (x_i - x_j)^2 over all pairs = 2 var
X = load_breast_cancer().data
v = X.var(axis=0)
pair = np.array([((c[:, None] - c[None]) ** 2).mean() for c in X.T])
N["pair_identity_maxerr"] = float(np.abs(pair / (2 * v) - 1).max())
N["share_largest_var"] = float(v.max() / v.sum())
N["share_second"] = float(np.sort(v)[-2] / v.sum())
N["largest_feature"] = load_breast_cancer().feature_names[v.argmax()]
N["second_feature"] = load_breast_cancer().feature_names[np.argsort(v)[-2]]
N["n_features_share_below_1e-4"] = int((v / v.sum() < 1e-4).sum())
# 2. Example 49.4-2: Box-Cox profile log-likelihood for 1, 2, 4, 8
x = np.array([1.0, 2.0, 4.0, 8.0]); lg = np.log(x).sum(); n = len(x)
def llf(l):
    z = np.log(x) if l == 0 else (x ** l - 1) / l
    return z, z.var(), -n / 2 * np.log(z.var()) + (l - 1) * lg
ex = {}
for l in (1.0, 0.5, 0.0, -0.5):
    z, s2, ll = llf(l); ex[str(l)] = {"psi": z.round(4).tolist(), "var": round(float(s2), 4), "ll": round(float(ll), 4)}
N["ex2"] = ex; N["ex2_sumlog"] = float(lg)
pt = PowerTransformer(method="box-cox", standardize=False).fit(x.reshape(-1, 1))
N["ex2_lambda_hat"] = float(pt.lambdas_[0]); N["ex2_ll_hat"] = float(llf(pt.lambdas_[0])[2])
# 3. quantile and robust constants
q = QuantileTransformer(n_quantiles=5).fit(np.array([[1.], [2.], [3.], [4.], [10.]]))
qn = QuantileTransformer(n_quantiles=5, output_distribution="normal").fit(np.array([[1.], [2.], [3.], [4.], [10.]]))
N["q_new_3.5_uniform"] = float(q.transform([[3.5]])[0, 0]); N["q_new_3.5_normal"] = float(qn.transform([[3.5]])[0, 0])
N["q_new_20_uniform"] = float(q.transform([[20.]])[0, 0]); N["q_new_0_uniform"] = float(q.transform([[0.]])[0, 0])
N["clip"] = float(ndtri(1e-7 - np.spacing(1))); N["clip_max_transform"] = float(qn.transform([[10.]])[0, 0])
N["unit_variance_divisor"] = float(ndtri(0.75) - ndtri(0.25))
g = np.random.RandomState(0).normal(size=100000, scale=2.0).reshape(-1, 1)
N["robust_scale_normal"] = float(RobustScaler().fit(g).scale_[0]); N["robust_scale_normal_uv"] = float(RobustScaler(unit_variance=True).fit(g).scale_[0])
# 4. messages
msgs = {}
def catch(name, f):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try: f(); msgs[name] = "no error; warnings: " + "; ".join(str(i.message)[:90] for i in w)
        except Exception as e: msgs[name] = type(e).__name__ + ": " + str(e)[:150]
S = sparse.csr_matrix(np.array([[0., 1.], [2., 0.], [0., 3.]]))
catch("std_sparse", lambda: StandardScaler().fit(S))
catch("minmax_sparse", lambda: MinMaxScaler().fit(S))
catch("boxcox_nonpos", lambda: PowerTransformer(method="box-cox").fit(np.array([[1.], [0.], [2.]])))
catch("nq_gt_n", lambda: QuantileTransformer(n_quantiles=1000).fit(np.arange(100.).reshape(-1, 1)))
catch("ft_inverse", lambda: FunctionTransformer(np.log1p, inverse_func=np.sqrt).fit(np.arange(1., 300.).reshape(-1, 1)))
N["messages"] = msgs
# 5. box-cox limit and kernel centerer
N["bc_limit_err"] = float(abs((3.0 ** 1e-9 - 1) / 1e-9 - np.log(3.0)))
Z = load_digits().data[:50] / 16.; K = linear_kernel(Z); J = np.ones((50, 50)) / 50
Kt = K - J @ K - K @ J + J @ K @ J
N["kc_formula_err"] = float(np.abs(KernelCenterer().fit_transform(K) - Kt).max())
Zc = Z - Z.mean(0); N["kc_linear_err"] = float(np.abs(Kt - Zc @ Zc.T).max()); N["kc_rowsum"] = float(np.abs(Kt.sum(1)).max())
N["binarizer"] = Binarizer(threshold=2.0).fit_transform([[1., 2., 3.]]).tolist()

# 7. extra constants quoted in Sections 49.3 to 49.7
from scipy import stats
from sklearn.preprocessing import Normalizer
x5 = np.array([[1.], [2.], [3.], [4.], [10.]])
N["ddof1_sd"] = float(np.std(x5, ddof=1)); N["ddof0_sd"] = float(np.std(x5))
mm = MinMaxScaler(feature_range=(-1, 1)).fit(x5)
N["mm_pm1_scale"] = float(mm.scale_[0]); N["mm_pm1_min"] = float(mm.min_[0])
N["const_scale"] = float(StandardScaler().fit([[5.], [5.], [5.]]).scale_[0])
N["q_uniform_train"] = QuantileTransformer(n_quantiles=5).fit_transform(x5).ravel().tolist()
N["n_quantiles_default"] = QuantileTransformer().n_quantiles
N["normalizer_zero_row"] = Normalizer().fit_transform([[0., 0.], [3., -4.]]).tolist()
N["norms_3_m4"] = [7.0, 5.0, 4.0]
xx = np.array([1., 2., 4., 8., 3., 5.])
def prof(l, d):
    z = np.log(d) if l == 0 else (d ** l - 1) / l
    return -len(d) / 2 * np.log(z.var()) + (l - 1) * np.log(d).sum()
N["boxcox_llf_err"] = max(abs(prof(l, xx) - stats.boxcox_llf(l, xx)) for l in (-1., 0.5, 2.))
xs = np.array([-3., -1., 0.5, 2., 4.])
def prof_yj(l, d):
    z = stats.yeojohnson(d, lmbda=l)
    return -len(d) / 2 * np.log(z.var()) + (l - 1) * (np.sign(d) * np.log1p(np.abs(d))).sum()
N["yj_llf_err"] = max(abs(prof_yj(l, xs) - stats.yeojohnson_llf(l, xs)) for l in (-1., 0.5, 2.))

# 6. quantile comb: how many distinct references
N["subsample_default"] = QuantileTransformer().subsample
print(json.dumps(N, indent=1))
json.dump(N, open("verify/prose_numbers.json", "w"), indent=1)

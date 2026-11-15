# Verification of every API statement and number quoted in prose (not a book listing).
import json, warnings
import numpy as np
from scipy import linalg
from sklearn.covariance import ledoit_wolf_shrinkage, LedoitWolf, OAS, empirical_covariance
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
R = {}
def show(k, v): R[k] = v; print(k, '=', v)

# ---- toy 2-D
X = np.array([[1,1],[3,3],[1,2],[3,2],[4,2],[6,4],[5,2],[5,4]], float); y = np.array([0,0,0,0,1,1,1,1])
Sw = (np.cov(X[y==0].T, bias=True) + np.cov(X[y==1].T, bias=True)) / 2
d = np.array([3., 1.]); Si = np.linalg.inv(Sw)
show('toy_S0_sums', (np.cov(X[y==0].T, bias=True)*4).tolist()); show('toy_S1_sums', (np.cov(X[y==1].T, bias=True)*4).tolist())
show('toy_Sw', Sw.tolist()); show('toy_Sinv', Si.tolist()); show('toy_w', (Si@d).tolist())
show('toy_maha2', float(d@Si@d)); show('toy_maha', float(np.sqrt(d@Si@d)))
a = np.array([7., -3.]); J = lambda a: float((a@d)**2/(a@Sw@a))
show('toy_J_fisher', J(a)); show('toy_J_meandiff', J(d)); show('toy_J_x', J(np.array([1.,0]))); show('toy_J_y', J(np.array([0.,1])))
show('toy_proj_fisher_class0', (X[y==0]@a).tolist()); show('toy_proj_fisher_class1', (X[y==1]@a).tolist())
show('toy_proj_md_class0', (X[y==0]@d).tolist()); show('toy_proj_md_class1', (X[y==1]@d).tolist())
show('toy_aSa', float(a@Sw@a)); show('toy_unit_a', (a/np.sqrt(a@Sw@a)).tolist())
for s in ('svd','lsqr','eigen'):
    m = LDA(solver=s).fit(X, y); show('toy_'+s, [np.round(m.coef_,6).tolist(), np.round(m.intercept_,6).tolist()])
# quadratic QDA on toy 2D
q = QDA(store_covariance=True).fit(X, y); show('toy_qda_cov', [c.tolist() for c in q.covariance_])

# ---- 1-D seeds with sklearn
A = np.array([1.,3,3,3,3,5]); Bv = np.array([6.,7,7,8]); x1 = np.r_[A, Bv][:, None]; y1 = np.r_[np.zeros(6), np.ones(4)]
l = LDA().fit(x1, y1); show('seed_coef', float(l.coef_[0,0])); show('seed_int', float(l.intercept_[0])); show('seed_x_star', float(-l.intercept_[0]/l.coef_[0,0]))
le = LDA(priors=[.5,.5]).fit(x1, y1); show('seed_x_star_eq', float(-le.intercept_[0]/le.coef_[0,0]))
show('seed_pred_5.05', [int(l.predict([[5.05]])[0]), int(le.predict([[5.05]])[0])])
qq = QDA().fit(x1, y1); show('seed_qda_pred', {str(v): int(qq.predict([[v]])[0]) for v in (5.2, 5.5, 8, 12, 14)})
show('seed_lda_P_at_8', float(l.predict_proba([[8.]])[0,1])); 
show('seed_qda_P_at_8', float(qq.predict_proba([[8.]])[0,1])); show('seed_qda_P_at_14', float(qq.predict_proba([[14.]])[0,1]))
show('seed_qda_scalings', [s.tolist() for s in qq.scalings_])

# ---- API behaviours
w = load_wine(); Xw, yw = w.data, w.target
def err(f):
    try: f(); return 'no error'
    except Exception as e: return type(e).__name__ + ': ' + str(e)[:110]
show('svd_shrinkage', err(lambda: LDA(solver='svd', shrinkage=0.3).fit(Xw, yw)))
show('svd_estimator', err(lambda: LDA(solver='svd', covariance_estimator=OAS()).fit(Xw, yw)))
show('lsqr_transform', err(lambda: LDA(solver='lsqr').fit(Xw, yw).transform(Xw)))
show('ncomp_3', err(lambda: LDA(n_components=3).fit(Xw, yw)))
show('shrink_and_estimator', err(lambda: LDA(solver='lsqr', shrinkage=0.2, covariance_estimator=OAS()).fit(Xw, yw)))
show('estimator_ok', [round(LDA(solver='lsqr', covariance_estimator=oas).fit(Xw, yw).score(Xw, yw),4) for oas in (OAS(), LedoitWolf())])
m = LDA().fit(Xw, yw); show('svd_no_cov_attr', hasattr(m, 'covariance_')); show('svd_has_xbar', hasattr(m, 'xbar_'))
show('lsqr_attrs', [hasattr(LDA(solver='lsqr').fit(Xw, yw), a_) for a_ in ('scalings_','explained_variance_ratio_','xbar_')])
show('eigen_attrs', [hasattr(LDA(solver='eigen').fit(Xw, yw), a_) for a_ in ('scalings_','explained_variance_ratio_','xbar_')])
show('wine_ratio_svd', np.round(LDA().fit(Xw, yw).explained_variance_ratio_, 4).tolist())
show('wine_ratio_eigen', np.round(LDA(solver='eigen').fit(Xw, yw).explained_variance_ratio_, 4).tolist())
show('wine_transform_shape', [LDA().fit(Xw, yw).transform(Xw).shape, LDA(n_components=1).fit(Xw, yw).transform(Xw).shape])
show('wine_coef_shape', list(LDA().fit(Xw, yw).coef_.shape)); show('wine_scalings_shape', list(LDA().fit(Xw, yw).scalings_.shape))
Z = LDA().fit(Xw, yw).transform(Xw); 
show('wine_unit_within', np.round(np.diag(np.cov(np.vstack([Z[yw==k]-Z[yw==k].mean(0) for k in range(3)]).T, bias=True)),4).tolist())
show('wine_centroids_proj', [Z[yw==k].mean(0).round(3).tolist() for k in range(3)])
# covariance_ = weighted biased
lq = LDA(solver='lsqr').fit(Xw, yw); pri = lq.priors_
Sw_w = sum(p*np.cov(Xw[yw==k].T, bias=True) for p, k in zip(pri, range(3))); show('cov_is_weighted_biased', bool(np.allclose(lq.covariance_, Sw_w)))
Sp = sum((np.sum(yw==k)-1)*np.cov(Xw[yw==k].T) for k in range(3))/(len(yw)-3); show('pooled_unbiased_ratio', float(np.trace(Sp)/np.trace(Sw_w)))
show('ratio_expected', float(len(yw)/(len(yw)-3)))
# float shrinkage
a_ = 0.3; ls = LDA(solver='lsqr', shrinkage=a_).fit(Xw, yw); p_ = Xw.shape[1]
show('shrink_formula', bool(np.allclose(ls.covariance_, (1-a_)*Sw_w + a_*np.trace(Sw_w)/p_*np.eye(p_))))
# auto
la = LDA(solver='lsqr', shrinkage='auto').fit(Xw, yw)
chk = np.zeros((p_,p_)); aa = []
for k in range(3):
    Xk = Xw[yw==k]; sc = StandardScaler().fit(Xk); Zk = sc.transform(Xk); a_k = ledoit_wolf_shrinkage(Zk)
    S = empirical_covariance(Xk); chk += pri[k]*((1-a_k)*S + a_k*np.diag(np.diag(S))); aa.append(round(float(a_k),4))
show('auto_formula', bool(np.allclose(la.covariance_, chk))); show('auto_coeffs_wine', aa)
# QDA reg_param
Xb = X; 
q0 = QDA(store_covariance=True).fit(Xw, yw); q5 = QDA(reg_param=0.5, store_covariance=True).fit(Xw, yw)
Sk = np.cov(Xw[yw==0].T, bias=True); show('qda_cov_biased', bool(np.allclose(q0.covariance_[0], Sk)))
show('qda_reg_svd_changes', bool(not np.allclose(q0.decision_function(Xw), q5.decision_function(Xw))))
# eigenvalue form check of reg: scalings = (1-r)*S2 + r  -> cov = (1-r)Sk + r I (as stored)
show('qda_reg_cov', bool(np.allclose(q5.covariance_[0], 0.5*Sk + 0.5*np.eye(p_))))
e0 = QDA(solver='eigen', store_covariance=True).fit(Xw, yw); e5 = QDA(solver='eigen', reg_param=0.5).fit(Xw, yw)
show('qda_eigen_ignores_reg', bool(np.allclose(e0.decision_function(Xw), e5.decision_function(Xw))))
show('qda_eigen_eq_svd', bool(np.allclose(e0.decision_function(Xw), q0.decision_function(Xw))))
show('qda_eigen_shrink_runs', round(QDA(solver='eigen', shrinkage='auto').fit(Xw, yw).score(Xw, yw), 4))
show('qda_svd_shrink', err(lambda: QDA(shrinkage=0.2).fit(Xw, yw)))
# QDA small class
Xs, ys = Xw[:70], yw[:70]; show('qda_small_class', err(lambda: QDA().fit(Xs[(ys==0)|(np.arange(70)>=58)][:20], ys[(ys==0)|(np.arange(70)>=58)][:20])))
rng = np.random.default_rng(0); Xr = rng.normal(size=(30, 10)); yr = np.r_[np.zeros(20), np.ones(10)]
show('qda_nk_le_p', err(lambda: QDA().fit(Xr, yr)))
show('qda_nk_le_p_reg', round(QDA(reg_param=0.2).fit(Xr, yr).score(Xr, yr), 4))
show('qda_default_wine_raw', err(lambda: QDA().fit(Xw, yw)))
# priors warnings
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always'); LDA(priors=[0.5, 0.3, 0.1]).fit(Xw, yw); show('priors_nosum', [str(r.message) for r in rec])
# parameter counts
cnt = lambda K, p, quad: K*p + (K if quad else 1)*p*(p+1)//2 + K - 1
show('count_lda_cancer', cnt(2,30,False)); show('count_qda_cancer', cnt(2,30,True)); show('count_lda_wine', cnt(3,13,False)); show('count_qda_wine', cnt(3,13,True)); show('count_lr_cancer', 31)
# wine CV comparison
cvs = {}
for name, mdl in [('LDA', LDA()), ('QDA', QDA()), ('LDA auto', LDA(solver='lsqr', shrinkage='auto')), ('logistic', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))]:
    cvs[name] = round(float(cross_val_score(mdl, Xw, yw, cv=5).mean()), 4)
show('wine_cv', cvs)
show('wine_train_acc_lda', round(LDA().fit(Xw, yw).score(Xw, yw), 4))
# LDA vs LR direction on breast cancer
dc = load_breast_cancer(); Xt, Xe, yt, ye = train_test_split(dc.data, dc.target, test_size=0.25, random_state=0, stratify=dc.target)
sc = StandardScaler().fit(Xt); l1 = LDA().fit(sc.transform(Xt), yt); l2 = LogisticRegression().fit(sc.transform(Xt), yt)
c1, c2 = l1.coef_[0], l2.coef_[0]; show('cancer_cos_lda_lr', round(float(c1@c2/np.linalg.norm(c1)/np.linalg.norm(c2)), 4))
show('cancer_norms', [round(float(np.linalg.norm(c1)),3), round(float(np.linalg.norm(c2)),3)])
show('cancer_prior', [round(float(v),4) for v in l1.priors_])
sl = l1.decision_function(sc.transform(Xe)); show('cancer_lda_score_range', [round(float(sl.min()),2), round(float(sl.max()),2)])
sr = l2.decision_function(sc.transform(Xe)); show('cancer_lr_score_range', [round(float(sr.min()),2), round(float(sr.max()),2)])
# svd vs lsqr on cancer coefficients equal
show('cancer_svd_lsqr_equal', bool(np.allclose(LDA().fit(Xt, yt).coef_, LDA(solver='lsqr').fit(Xt, yt).coef_, atol=1e-6)))
# feature-scaling invariance
sc_ = StandardScaler().fit(Xw); Xs_ = sc_.transform(Xw)
def same(m1, m2): return bool(np.allclose(m1.fit(Xw, yw).predict_proba(Xw), m2.fit(Xs_, yw).predict_proba(Xs_), atol=1e-6))
show('scale_inv_lda_svd', same(LDA(), LDA())); show('scale_inv_lda_lsqr', same(LDA(solver='lsqr'), LDA(solver='lsqr')))
show('scale_inv_lda_auto', same(LDA(solver='lsqr', shrinkage='auto'), LDA(solver='lsqr', shrinkage='auto')))
show('scale_inv_lda_fixed', same(LDA(solver='lsqr', shrinkage=0.3), LDA(solver='lsqr', shrinkage=0.3)))
show('scale_inv_qda', same(QDA(), QDA())); show('scale_inv_qda_reg', same(QDA(reg_param=0.3), QDA(reg_param=0.3)))
show('scale_fixed_label_changes', int((LDA(solver='lsqr', shrinkage=0.3).fit(Xw, yw).predict(Xw) != LDA(solver='lsqr', shrinkage=0.3).fit(Xs_, yw).predict(Xs_)).sum()))
show('scale_qda_reg_label_changes', int((QDA(reg_param=0.3).fit(Xw, yw).predict(Xw) != QDA(reg_param=0.3).fit(Xs_, yw).predict(Xs_)).sum()))
json.dump(R, open('../verify/facts.json', 'w'), indent=1, default=str)

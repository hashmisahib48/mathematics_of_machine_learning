# Numbers quoted in the prose of Chapter 19 (no figures). Run: python3 fig/facts.py
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..')
import numpy as np
from sklearn.datasets import load_digits
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
F={}
msgs=["free prize now","win free prize","win free now now","meeting report","lunch meeting now","report lunch meeting"]
y=np.array([1,1,1,0,0,0]); vec=CountVectorizer(); X=vec.fit_transform(msgs); new=vec.transform(["free meeting now","win prize now"])
# unsmoothed
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    m0=MultinomialNB(alpha=0,force_alpha=True).fit(X,y)
    F['alpha0_joint_free_meeting_now']=[float(v) for v in np.exp(m0.predict_joint_log_proba(new[:1]))[0]]
    F['alpha0_joint_win_prize_now']=[float(v) for v in np.exp(m0.predict_joint_log_proba(new[1:]))[0]]
    F['alpha0_proba_free_meeting_now']=[str(v) for v in m0.predict_proba(new[:1])[0]]
# complement = multinomial, two classes, equal priors
mn=MultinomialNB().fit(X,y); cn=ComplementNB().fit(X,y)
F['comp_equals_mult_2class']=bool(np.allclose(mn.predict_proba(new),cn.predict_proba(new)))
cn2=ComplementNB(class_prior=[0.9,0.1]).fit(X,y); F['comp_prior_ignored_2class']=bool(np.allclose(cn.predict_proba(new),cn2.predict_proba(new)))
mn2=MultinomialNB(class_prior=[0.9,0.1]).fit(X,y); F['mult_prior_used_proba_spam']=round(float(mn2.predict_proba(new[:1])[0,1]),4)
F['coef_attr_multinomial']=hasattr(mn,'coef_')
# linear rule: multinomial two-class
w=mn.feature_log_prob_[1]-mn.feature_log_prob_[0]; b=mn.class_log_prior_[1]-mn.class_log_prior_[0]
xx=new.toarray(); lo=xx@w+b; pp=mn.predict_proba(new)[:,1]
F['mult_w']=[round(float(v),4) for v in w]; F['mult_b']=round(float(b),4); F['mult_logodds_hand']=[round(float(v),4) for v in lo]
F['mult_logodds_sk']=[round(float(v),4) for v in np.log(pp/(1-pp))]
# bernoulli linear
bn=BernoulliNB().fit(X,y); th=np.exp(bn.feature_log_prob_); xb=(xx>0).astype(float)
wb=np.log(th[1]*(1-th[0])/(th[0]*(1-th[1]))); bb=bn.class_log_prior_[1]-bn.class_log_prior_[0]+np.log((1-th[1])/(1-th[0])).sum()
pb=bn.predict_proba(new)[:,1]; F['bern_logodds_hand']=[round(float(v),4) for v in xb@wb+bb]; F['bern_logodds_sk']=[round(float(v),4) for v in np.log(pb/(1-pb))]
F['bern_theta_spam']=[round(float(v),4) for v in th[1]]; F['bern_theta_ham']=[round(float(v),4) for v in th[0]]
# gaussian toy
Xg=np.array([[1,2],[2,4],[3,3],[4,5],[6,9],[8,7]],float); yg=np.array([0,0,0,1,1,1]); g=GaussianNB().fit(Xg,yg)
q=np.array([[3.,6.]]); F['toy_dist_sq']=[float(((q-g.theta_[k])**2).sum()) for k in (0,1)]
F['toy_jll']=[round(float(v),4) for v in g.predict_joint_log_proba(q)[0]]; F['toy_proba']=[round(float(v),4) for v in g.predict_proba(q)[0]]
# decomposed terms for the example
v=g.var_; F['toy_logdens_parts']=[[round(float(-0.5*np.log(2*np.pi*v[k,j])),4) for j in range(2)] for k in range(2)]
F['toy_quad_parts']=[[round(float((q[0,j]-g.theta_[k,j])**2/(2*v[k,j])),4) for j in range(2)] for k in range(2)]
# negative values error
Xn=StandardScaler().fit_transform(load_digits().data[:50])
try: MultinomialNB().fit(Xn,load_digits().target[:50])
except ValueError as e: F['neg_error']=str(e)
# underflow
F['underflow_prod']=float(np.prod(np.full(1000,0.01))); F['underflow_logsum']=round(float(1000*np.log(0.01)),2)
# parameter counts
F['params']={str(p):[2*(2**p-1)+1,2*p+1] for p in (5,10,20,30)}
# digits: complement ignores prior; scaling invariance; logistic
d=load_digits(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0,stratify=d.target)
c1=ComplementNB().fit(Xtr,ytr); c2=ComplementNB(class_prior=np.r_[0.91,np.full(9,0.01)]).fit(Xtr,ytr); F['comp_digits_prior_ignored']=bool((c1.predict(Xte)==c2.predict(Xte)).all())
mm=MultinomialNB(class_prior=np.r_[0.91,np.full(9,0.01)]).fit(Xtr,ytr); F['mult_digits_prior_changes_acc']=round(float(mm.score(Xte,yte)),4)
cn_norm=ComplementNB(norm=True).fit(Xtr,ytr); F['comp_digits_norm_acc']=round(float(cn_norm.score(Xte,yte)),4)
from sklearn.datasets import load_iris
Xi,yi=load_iris(return_X_y=True); sc=np.array([100,0.01,5,1.0]); pa=GaussianNB(var_smoothing=0).fit(Xi,yi).predict_proba(Xi); pb2=GaussianNB(var_smoothing=0).fit(Xi*sc,yi).predict_proba(Xi*sc)
F['gauss_scale_invariant_iris_vs0']=bool(np.allclose(pa,pb2)); F['gauss_scale_maxdiff']=float(np.abs(pa-pb2).max())
F['const_pixels']=int((Xtr.std(0)==0).sum())
lr=LogisticRegression(max_iter=2000).fit(Xtr,ytr); F['logreg_digits_test']=round(float(lr.score(Xte,yte)),4)
# gaussian on digits: var_smoothing zero
with warnings.catch_warnings(record=True) as w_:
    warnings.simplefilter('always'); g0=GaussianNB(var_smoothing=0).fit(Xtr,ytr); F['vs0_acc']=round(float(g0.score(Xte,yte)),4)
    F['vs0_warnings']=sorted({str(x.category.__name__) for x in w_})
# partial_fit multinomial
a=MultinomialNB(); a.partial_fit(Xtr[:700],ytr[:700],classes=np.arange(10)); a.partial_fit(Xtr[700:],ytr[700:]); f=MultinomialNB().fit(Xtr,ytr)
F['mult_partial_equal']=bool(np.allclose(a.feature_log_prob_,f.feature_log_prob_) and np.allclose(a.class_log_prior_,f.class_log_prior_))
json.dump(F,open('verify/facts.json','w'),indent=1); print(json.dumps(F,indent=1))

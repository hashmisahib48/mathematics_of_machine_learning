# Numbers quoted in the worked examples of Chapter 14. Run: python3 fig/exnum.py
import os,json,numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..')
x=np.arange(5.);y=np.array([1.,2,2,4,6]);X=np.c_[np.ones(5),x]
d=lambda y,m:2*(y*np.log(y/m)-(y-m))
N={}
b0=np.array([np.log(3),0.]); mu0=np.exp(X@b0)
g0=X.T@(y-mu0); H0=X.T@(mu0[:,None]*X); dl0=np.linalg.solve(H0,g0)
N['step1']={'b0':b0[0],'mu':mu0.tolist(),'y_minus_mu':(y-mu0).tolist(),'g':g0.tolist(),'H':H0.tolist(),'delta':dl0.tolist(),'D':float(d(y,mu0).sum())}
b1=b0+dl0; eta1=X@b1; mu1=np.exp(eta1); g1=X.T@(y-mu1); H1=X.T@(mu1[:,None]*X); dl1=np.linalg.solve(H1,g1); b2=b1+dl1
N['step2']={'beta1':b1.tolist(),'eta':eta1.tolist(),'mu':mu1.tolist(),'y_minus_mu':(y-mu1).tolist(),'x_y_minus_mu':(x*(y-mu1)).tolist(),'mux':(mu1*x).tolist(),'mux2':(mu1*x*x).tolist(),
 'sum_mu':float(mu1.sum()),'sum_mux':float((mu1*x).sum()),'sum_mux2':float((mu1*x*x).sum()),'g':g1.tolist(),'H':H1.tolist(),'delta':dl1.tolist(),'beta2':b2.tolist(),'D1':float(d(y,mu1).sum())}
mu2=np.exp(X@b2); N['step2']['D2']=float(d(y,mu2).sum())
# converged
b=b2.copy()
for k in range(30):
    mu=np.exp(X@b); b=b+np.linalg.solve(X.T@(mu[:,None]*X),X.T@(y-mu))
mu=np.exp(X@b); Dnull=d(y,np.full(5,y.mean()))
N['fit']={'beta':b.tolist(),'mu':mu.tolist(),'d_i':d(y,mu).tolist(),'D':float(d(y,mu).sum()),'D_null_i':Dnull.tolist(),'D_null':float(Dnull.sum()),'D2':float(1-d(y,mu).sum()/Dnull.sum()),
  'expw':float(np.exp(b[1])),'e4w':float(np.exp(4*b[1])),'pred5':float(np.exp(b[0]+5*b[1])),'pct':float(100*(np.exp(b[1])-1))}
# unit deviances
dg=lambda y,m:2*((y-m)/m-np.log(y/m))
N['units']={}
for (yy,mm) in [(4.,2.),(40.,20.),(4.,8.),(2.,4.),(20.,40.)]:
    N['units'][f'{yy:g},{mm:g}']={'sq':(yy-mm)**2,'pois':float(d(yy,mm)),'gam':float(dg(yy,mm))}
N['pt3']={'ln_ratio':float(np.log(2/mu[2])),'term':float(2*(2*np.log(2/mu[2])-(2-mu[2])))}
# exposure
Nn,e,lam=3.,2.,1.
N['expo']={'count_dev':float(d(Nn,e*lam)),'rate_dev_times_e':float(e*d(Nn/e,lam))}
# OLS on toy for contrast
c=np.polyfit(x,y,1); N['ols']={'w':float(c[0]),'b':float(c[1]),'at_minus1':float(c[1]-c[0]),'at_minus2':float(c[1]-2*c[0])}
N['ols']['zero_at']=float(-c[1]/c[0]); N['fit']['glm_at_minus1']=float(np.exp(b[0]-b[1]))
# IRLS identity: Newton step equals weighted least squares on the working response
b_=np.array([np.log(3),0.]); mu_=np.exp(X@b_); z=X@b_+(y-mu_)/mu_; Mm=np.diag(mu_)
irls=np.linalg.solve(X.T@Mm@X,X.T@Mm@z); newt=b_+np.linalg.solve(X.T@Mm@X,X.T@(y-mu_)); N['irls']={'diff':float(np.abs(irls-newt).max()),'z':z.tolist()}
# compound Poisson-gamma: p=(k+2)/(k+1) check by simulation, Tweedie parametrisation p=1.5, phi=2, mu=3
rng=np.random.default_rng(0); p_,phi,mu_t=1.5,2.0,3.0; lam=mu_t**(2-p_)/(phi*(2-p_)); k=(2-p_)/(p_-1); th=phi*(p_-1)*mu_t**(p_-1)
cnt=rng.poisson(lam,400000); tot=np.array([rng.gamma(k*c_,th) if c_>0 else 0. for c_ in cnt[:200000]])
N['cpg']={'lambda':lam,'k':k,'scale':th,'mean_sim':float(tot.mean()),'var_sim':float(tot.var()),'var_formula':phi*mu_t**p_,'p_from_k':(k+2)/(k+1)}
os.makedirs('verify',exist_ok=True); json.dump(N,open('verify/example_numbers.json','w'),indent=1)
def r(v,k=4):
    return np.round(v,k).tolist() if not isinstance(v,float) else round(v,k)
print(json.dumps({k:{kk:r(vv) if not isinstance(vv,dict) else vv for kk,vv in v.items()} for k,v in N.items()},indent=0))

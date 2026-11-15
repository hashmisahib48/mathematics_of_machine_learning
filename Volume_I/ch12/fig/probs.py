# Problem figures and answers of Chapter 12. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.linear_model import BayesianRidge, ARDRegression, Ridge, LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5})
A={}; GL='#CFE8DA'
x=np.array([-3.,-1,1,3]); y=np.array([-2.,-2,1,3])
def small(w=2.0,h=1.3): return plt.subplots(figsize=(w,h),dpi=240)
def gx(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def sc(ax,xx,yy,s=26): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.7,zorder=6); gx(ax)
nd=lambda w,m,v: np.exp(-(w-m)**2/(2*v))/np.sqrt(2*np.pi*v)

# 12A.1 posterior for new data
x1=np.array([-2.,-1,1,2]); y1=np.array([-3.,-1,2,2]); a1,l1=2.0,4.0; Sxx=float(x1@x1); Sxy=float(x1@y1); pr=l1+a1*Sxx; m1=a1*Sxy/pr
rd=Ridge(alpha=l1/a1,fit_intercept=False).fit(x1[:,None],y1)
A['12A.1']={'Sxx':Sxx,'Sxy':Sxy,'precision':pr,'mean':round(m1,4),'sd':round(pr**-0.5,4),'ols':round(Sxy/Sxx,4),'ridge_alpha':l1/a1,'ridge_coef':round(float(rd.coef_[0]),4)}
fig,ax=small(); sc(ax,x1,y1,34); ax.plot([-3,3],[m1*-3,m1*3],color=GREEN,lw=1.5)
for a_,b_ in zip(x1,y1):
    dx,dy,ha={-2:(0.13,-0.6,'left'),-1:(0.13,-0.6,'left'),1:(-0.1,0.5,'right'),2:(0.0,-1.0,'center')}[int(a_)]
    ax.text(a_+dx,b_+dy,f'({a_:.0f}, {b_:.0f})'.replace('-','\u2212'),fontsize=9,color=GREY,ha=ha)
ax.set_xlim(-3,3); ax.set_ylim(-4.5,4.6); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); save(fig,'p_12A1.png')

# 12A.2 predictive sd = 2 noise sd
V=1/25.; xstar=float(np.sqrt(3/V)); A['12A.2']={'xstar':round(xstar,3),'sd_at_3':round(float(np.sqrt(1+9*V)),3)}
fig,ax=small(); xs=np.linspace(-10,10,200); ax.plot(xs,np.sqrt(1+xs**2*V),color=GREEN,lw=1.6); ax.axhline(2,color=PUR,lw=1,ls='--'); ax.axhline(1,color=GREY,lw=1,ls=':')
ax.plot([xstar],[2],'o',color=PUR,ms=4.5); ax.text(-9.6,2.15,'twice the noise',fontsize=9,color=PUR,va='bottom'); ax.set_ylim(0,3.1); ax.set_xlim(-10,10); ax.set_xlabel('$x_*$'); ax.set_ylabel('predictive s.d.'); gx(ax); save(fig,'p_12A2.png')

# 12A.3 two pairs with the same ratio
res={}
for a_,l_ in ((1.,5.),(4.,20.)):
    pr_=l_+a_*20; res[f'{a_:g},{l_:g}']={'mean':round(a_*18/pr_,4),'sd':round(pr_**-0.5,4)}
A['12A.3']=res
fig,ax=small(); wg=np.linspace(0.2,1.3,300)
for (a_,l_),c,ls_ in (((1.,5.),GREEN,'-'),((4.,20.),OR,'--')): pr_=l_+a_*20; ax.plot(wg,nd(wg,a_*18/pr_,1/pr_),color=c,lw=1.6,ls=ls_)
ax.text(0.2,3.7,'$\\alpha$ = 4, $\\lambda$ = 20',color=OR,fontsize=9,va='top'); ax.text(1.28,1.3,'$\\alpha$ = 1,\n$\\lambda$ = 5',color=GREEN,fontsize=9,ha='right',va='center')
ax.set_xlabel('weight $w$'); ax.set_ylabel('posterior density'); ax.set_ylim(0,4.3); gx(ax); save(fig,'p_12A3.png')

# 12A.4 one update from alpha=2, lambda=1
a0,l0=2.0,1.0; pr=l0+a0*20; m_=a0*18/pr; g=a0*20/pr; sse=float(np.sum((y-m_*x)**2)); ln=g/m_**2; an=(4-g)/sse
A['12A.4']={'precision':pr,'m':round(m_,4),'gamma':round(g,4),'sse':round(sse,4),'lambda_new':round(ln,4),'alpha_new':round(an,4)}
fig,ax=small(); ax.plot([a0,an],[l0,ln],'-',color=PUR,lw=1); ax.annotate('',xy=(an,ln),xytext=(a0,l0),arrowprops=dict(arrowstyle='-|>',color=PUR,lw=1.2)); ax.plot(a0,l0,'s',color=NAVY,ms=5)
ax.plot(5/3,1.282,'*',color=GREEN,ms=10); ax.text(5/3+0.05,1.282+0.05,'converged',fontsize=9,color=GREEN,va='bottom'); ax.text(a0,l0-0.06,'start',fontsize=9,color=NAVY,va='top',ha='center')
ax.set_xlim(1.0,2.8); ax.set_ylim(0.3,2.2); ax.set_xlabel('noise precision $\\alpha$'); ax.set_ylabel('weight precision $\\lambda$'); gx(ax); save(fig,'p_12A4.png')

# 12B.1 log evidence method
from bayes_scratch import MyBayesRidge
class Mine(MyBayesRidge):
    def log_evidence(self,X,y):
        Xc=X-X.mean(0); yc=y-y.mean(); n,p=Xc.shape; al,la=self.alpha_,self.lambda_
        Am=la*np.eye(p)+al*Xc.T@Xc; m=al*np.linalg.solve(Am,Xc.T@yc)
        return 0.5*(p*np.log(la)+n*np.log(al)-n*np.log(2*np.pi)-np.linalg.slogdet(Am)[1]-al*np.sum((yc-Xc@m)**2)-la*m@m)
X=x[:,None]; mine=Mine().fit(X,y); z0=dict(alpha_1=0,alpha_2=0,lambda_1=0,lambda_2=0); b0=BayesianRidge(compute_score=True,**z0).fit(X,y)
A['12B.1']={'mine':round(mine.log_evidence(X,y),4),'sklearn_scores_last':round(float(b0.scores_[-1]),4),'sklearn_scores':[round(float(v),4) for v in b0.scores_]}
fig,ax=small(); ax.plot(np.arange(len(b0.scores_)),b0.scores_,'-o',color=PUR,ms=4,lw=1.3); ax.set_xlabel('iteration'); ax.set_ylabel('log evidence'); gx(ax); save(fig,'p_12B1.png')

# 12B.2 more data, narrower band
rows={}
fig,ax=small(); xs=np.linspace(-9,9,200)
for k,c in zip((1,4,16),(GREY,OR,GREEN)):
    Xk=np.tile(x,k)[:,None]; yk=np.tile(y,k); bk=BayesianRidge().fit(Xk,yk); mu,sd=bk.predict(np.array([[6.0]]),return_std=True)
    rows[str(k)]={'alpha_':round(float(bk.alpha_),3),'lambda_':round(float(bk.lambda_),3),'sd_w':round(float(np.sqrt(bk.sigma_[0,0])),4),'sd_pred6':round(float(sd[0]),4)}
    ax.plot(xs,bk.predict(xs[:,None],return_std=True)[1],color=c,lw=1.5,label=f'{k} $\\times$ data')
A['12B.2']=rows; ax.set_xlabel('$x_*$'); ax.set_ylabel('predictive s.d.'); ax.legend(fontsize=9,loc='upper center'); ax.set_ylim(0,3.8); gx(ax); save(fig,'p_12B2.png')

# 12B.3 ARD and a duplicated feature
rng=np.random.default_rng(1); n=50; u=rng.normal(size=n); Xd=np.c_[u,u,rng.normal(size=(n,3))]; yd=2*u+rng.normal(0,0.5,n)
bd=BayesianRidge().fit(Xd,yd); ad=ARDRegression().fit(Xd,yd)
A['12B.3']={'BR':[round(float(v),3) for v in bd.coef_],'ARD':[round(float(v),3) for v in ad.coef_],'ARD_lambda':[float(f'{v:.3g}') for v in ad.lambda_]}
fig,ax=small(); jj=np.arange(5); ax.bar(jj-0.18,bd.coef_,width=0.34,color=OR,label='BayesianRidge'); ax.bar(jj+0.18,ad.coef_,width=0.34,color=GREEN,label='ARDRegression')
ax.axhline(0,color=GREY,lw=0.6); ax.set_xticks(jj); ax.set_xticklabels(['$x_1$','$x_1$ copy','$z_1$','$z_2$','$z_3$'],fontsize=9); ax.set_ylabel('weight'); ax.set_ylim(-0.3,2.4); ax.legend(fontsize=9,loc='upper right'); gx(ax); save(fig,'p_12B3.png')

# 12C.1 the floor 1/sqrt(alpha)
fig,ax=small(); xs=np.linspace(-6,6,200); bk=BayesianRidge().fit(np.tile(x,64)[:,None],np.tile(y,64)); mu,sd=bk.predict(xs[:,None],return_std=True)
ax.fill_between(xs,mu-2*sd,mu+2*sd,color=GL,lw=0); ax.plot(xs,mu,color=GREEN,lw=1.4); ax.plot(xs,mu-2/np.sqrt(bk.alpha_),color=GREY,lw=1,ls='--'); ax.plot(xs,mu+2/np.sqrt(bk.alpha_),color=GREY,lw=1,ls='--'); sc(ax,x,y,s=18)
ax.set_xlabel('$x_*$'); ax.set_ylabel('$y$'); ax.set_ylim(-9,9); save(fig,'p_12C1.png'); A['12C.1']={'alpha_256pts':round(float(bk.alpha_),3),'floor':round(float(bk.alpha_**-0.5),3),'min_sd':round(float(sd.min()),3)}

# 12C.2 evidence and cross-validation, diabetes
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0); fn=json.load(open('verify/fig_numbers.json'))
rat=np.logspace(-3,1.6,24); cv=np.array([-cross_val_score(Ridge(alpha=r),Xtr,ytr,cv=5,scoring='neg_mean_squared_error').mean() for r in rat]); br=BayesianRidge().fit(Xtr,ytr); rb=br.lambda_/br.alpha_
fig,ax=small(2.0,1.3); ax.semilogx(rat,(cv-cv.min())/(cv.max()-cv.min()),'-o',color=GREY,ms=2.8,lw=1.2); ax.axvline(rb,color=GREEN,lw=1.4); ax.set_xlabel('$\\lambda/\\alpha$'); ax.set_ylabel('CV error (scaled)'); ax.text(rb*1.3,0.55,'evidence',color=GREEN,fontsize=9); gx(ax); save(fig,'p_12C2.png')
A['12C.2']={'ratio_evidence':float(rb),'ratio_cv':float(fn['valid']['ratio_best_cv'])}

# 12C.3 heteroscedastic noise breaks the band
rng=np.random.default_rng(5); nn=300; xh=rng.uniform(0,10,nn); yh=1+0.8*xh+rng.normal(0,0.2+0.35*xh,nn); bh=BayesianRidge().fit(xh[:,None],yh)
mu,sd=bh.predict(xh[:,None],return_std=True); cov=float(np.mean(np.abs(yh-mu)<2*sd)); lo=float(np.mean((np.abs(yh-mu)<2*sd)[xh<2.5])); hi=float(np.mean((np.abs(yh-mu)<2*sd)[xh>7.5]))
A['12C.3']={'coverage2sd':round(cov,3),'coverage_x_lt_2.5':round(lo,3),'coverage_x_gt_7.5':round(hi,3)}
fig,ax=small(); xs=np.linspace(0,10,100); m_,s_=bh.predict(xs[:,None],return_std=True); ax.fill_between(xs,m_-2*s_,m_+2*s_,color=GL,lw=0); ax.scatter(xh,yh,s=3,color=BLUE,lw=0,alpha=0.7); ax.plot(xs,m_,color=GREEN,lw=1.4); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); gx(ax); save(fig,'p_12C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A,indent=0))

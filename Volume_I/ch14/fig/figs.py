# All main figures of Chapter 14. Run: python3 fig/figs.py   (PNGs into ch14/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
from scipy.stats import poisson, gamma as gam
from sklearn.linear_model import PoissonRegressor, TweedieRegressor, LinearRegression, GammaRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import train_test_split
from portfolio import make_portfolio, TRUE
NUM={}
def ax_(ax,xl,yl): ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx,yy,s=30,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.7,zorder=5,**k)
x5=np.arange(5.); y5=np.array([1.,2,2,4,6])
pm=PoissonRegressor(alpha=0,tol=1e-10).fit(x5[:,None],y5); b5,w5=float(pm.intercept_),float(pm.coef_[0])
om=LinearRegression().fit(x5[:,None],y5)
dev=lambda y,m:2*(y*np.log(np.where(y>0,y,1)/m)-(y-m))

# ---- 14.0-1 opener: counts that fall with x; OLS goes negative
rng=np.random.default_rng(4); xs=np.repeat(np.arange(0,9),5)+rng.uniform(-0.25,0.25,45); mu_t=np.exp(2.3-0.42*xs); ys=rng.poisson(mu_t)
pg=PoissonRegressor(alpha=0,tol=1e-10).fit(xs[:,None],ys); ol=LinearRegression().fit(xs[:,None],ys)
fig,ax=plt.subplots(figsize=(4.4,1.6),dpi=240); xx=np.linspace(0,10,200)
ax.axhspan(-3.2,0,color=PINK,zorder=0); ax.axhline(0,color=RED,lw=0.8)
ax.plot(xx,ol.predict(xx[:,None]),color=OR,lw=1.8,ls='--',zorder=3,label='least squares'); ax.plot(xx,pg.predict(xx[:,None]),color=GREEN,lw=2,zorder=4,label='Poisson GLM (log link)')
ys_j=ys+rng.uniform(-0.12,0.12,45)*0; pts(ax,xs,ys,s=22)
ax.text(0.3,-1.6,'impossible: negative counts',color=RED,fontsize=9,ha='left',va='center')
ax.legend(fontsize=9,loc='upper right',frameon=False)
ax.set_xlim(0,10); ax.set_ylim(-3.2,12); ax_(ax,'feature $x$','count $y$'); save(fig,'f1400.png')
NUM['opener']={'ols_at_10':round(float(ol.predict([[10.]])[0]),2),'glm_at_10':round(float(pg.predict([[10.]])[0]),2),'ols_zero_at':round(float(-ol.intercept_/ol.coef_[0]),2)}

# ---- 14.1-1 the crack data
fig,ax=plt.subplots(figsize=(3.3,2.1),dpi=240); xx=np.linspace(-2,6,200)
ax.axhspan(-3,0,color=PINK,zorder=0); ax.axhline(0,color=RED,lw=0.8)
ax.plot(xx,om.intercept_+om.coef_[0]*xx,color=OR,lw=1.6,ls='--',label='least squares'); ax.plot(xx,np.exp(b5+w5*xx),color=GREEN,lw=2,label='Poisson fit')
pts(ax,x5,y5,s=42); ax.legend(fontsize=8.5,loc='upper left',handlelength=2.2)
for a,b_ in zip(x5,y5): ax.text(a+(-0.38 if a==4 else 0.12),b_+(0.35 if a==4 else -0.95),f'{b_:.0f}',fontsize=8.5,color=GREY)
ax.set_xlim(-2,6); ax.set_ylim(-3,10.5); ax_(ax,'load cycles $x$ ($10^5$)','cracks $y$'); save(fig,'f1411.png')

# ---- 14.2-1 spread grows with the mean
fig,axs=plt.subplots(1,2,figsize=(4.9,2.05),dpi=240)
ax=axs[0]
for mu,c,off,mk in [(2,'#9DB9E5',-0.28,'o'),(8,BLUE,0,'s'),(20,NAVY,0.28,'^')]:
    k=np.arange(0,40); ax.vlines(k+off,0,poisson.pmf(k,mu),color=c,lw=1.3); ax.plot(k+off,poisson.pmf(k,mu),mk,color=c,ms=2.6)
for mu,xt,yt in [(2,3.3,0.265),(8,10.5,0.15),(20,24,0.1)]: ax.text(xt,yt,f'$\\mu$ = {mu}',fontsize=8.5,color=NAVY)
ax.set_xlim(-1,36); ax.set_ylim(0,0.31); ax_(ax,'count $y$','probability'); ax.set_title('Poisson: variance = mean',fontsize=9,color=NAVY)
ax=axs[1]; yy=np.linspace(0.01,40,800); k=3.0
for mu,c,ls in [(2,'#9DB9E5','-'),(8,BLUE,'--'),(20,NAVY,'-.')]:
    ax.plot(yy,gam.pdf(yy,k,scale=mu/k),color=c,lw=1.6,ls=ls)
for mu,xt,yt in [(2,3.4,0.38),(8,9.0,0.115),(20,24,0.06)]: ax.text(xt,yt,f'$\\mu$ = {mu}',fontsize=8.5,color=NAVY)
ax.set_xlim(0,38); ax.set_ylim(0,0.75); ax_(ax,'amount $y$','density'); ax.set_title('Gamma: s.d. = 0.58 $\\mu$',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1421.png')
NUM['gamma_cv']=round(float(1/np.sqrt(k)),3); NUM['pint']={str(m_):[int(v) for v in poisson.interval(0.95,m_)] for m_ in (2,8,20)}

# ---- 14.2-2 the log link: straight in eta, multiplicative in mu
fig,axs=plt.subplots(1,2,figsize=(4.9,2.0),dpi=240); xx=np.linspace(-0.3,4.4,100)
ax=axs[0]; ax.plot(xx,b5+w5*xx,color=GREEN,lw=1.8); ax.plot(x5,b5+w5*x5,'o',color=GREEN,ms=4.5)
for a in x5[:-1]: ax.annotate('',xy=(a+1,b5+w5*a),xytext=(a,b5+w5*a),arrowprops=dict(arrowstyle='-',color=PUR,lw=0.8)); ax.annotate('',xy=(a+1,b5+w5*(a+1)),xytext=(a+1,b5+w5*a),arrowprops=dict(arrowstyle='-',color=PUR,lw=0.8))
ax.text(2.6,b5+0.02,'each step: $+w$ = +%.2f'%w5,fontsize=8.5,color=PUR,va='top'); ax_(ax,'$x$','linear predictor $\\eta=\\ln\\mu$'); ax.set_title('straight line in $\\eta$',fontsize=9,color=NAVY); ax.set_ylim(-0.4,2.1)
ax=axs[1]; ax.plot(xx,np.exp(b5+w5*xx),color=GREEN,lw=1.8); mus=np.exp(b5+w5*x5); ax.plot(x5,mus,'o',color=GREEN,ms=4.5)
for a in range(4): ax.text(a+0.62,np.exp(b5+w5*(a+0.5))-0.85,f'×{np.exp(w5):.2f}',fontsize=8.5,color=PUR,ha='center',va='center')
ax_(ax,'$x$','mean $\\mu$'); ax.set_title('curve in $\\mu$: constant ratio',fontsize=9,color=NAVY); ax.set_ylim(0,7.4)
fig.tight_layout(w_pad=1.0); save(fig,'f1422.png')

# ---- 14.3-1 deviance curves
y0=4.0; mu=np.linspace(0.6,12,400)
def dtw(y,m,p):
    if p==0: return (y-m)**2
    if p==1: return 2*(y*np.log(y/m)-(y-m))
    if p==2: return 2*((y-m)/m-np.log(y/m))
    return 2*(y**(2-p)/((1-p)*(2-p))-y*m**(1-p)/(1-p)+m**(2-p)/(2-p))
fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240)
ax=axs[0]
for p,c,ls,lab in [(0,GREY,':','$p$=0  squared'),(1,BLUE,'-','$p$=1  Poisson'),(2,OR,'--','$p$=2  gamma'),(3,PUR,'-.','$p$=3')]:
    ax.plot(mu,dtw(y0,mu,p),color=c,lw=1.6,ls=ls,label=lab)
ax.axvline(y0,color=GREEN,lw=0.8); ax.set_ylim(0,10); ax.set_xlim(0,12); ax_(ax,'prediction $\\mu$','unit deviance $d(4,\\mu)$'); ax.legend(fontsize=7.6,loc='upper right',handlelength=2.0)
ax.set_title('cost of a miss, true $y$ = 4',fontsize=9,color=NAVY)
ax=axs[1]; eta=np.linspace(-0.4,3.4,300)
for p,c,ls,lab in [(1,BLUE,'-','Poisson'),(2,OR,'--','gamma')]:
    ax.plot(eta,0.5*dtw(y0,np.exp(eta),p),color=c,lw=1.6,ls=ls,label=lab)
ax.axvline(np.log(y0),color=GREEN,lw=0.8); ax.text(np.log(y0)+0.08,3.85,'$\\eta=\\ln 4$',fontsize=8.5,color=GREEN); ax.text(0.3,3.25,'Poisson',fontsize=8.5,color=BLUE); ax.text(2.75,0.3,'gamma',fontsize=8.5,color=OR)
ax.set_ylim(0,4.2); ax_(ax,'linear predictor $\\eta$','half deviance'); ax.set_title('one bowl in $\\eta$',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1431.png')

X=np.c_[np.ones(5),x5]
bs=[np.array([np.log(3),0.])]
for k in range(5):
    b_=bs[-1]; m=np.exp(X@b_); bs.append(b_+np.linalg.solve(X.T@(m[:,None]*X),X.T@(y5-m)))
Ds=[float(dev(y5,np.exp(X@b_)).sum()) for b_ in bs]; Dstar=Ds[-1]
NUM['newton']={'D':[round(v,5) for v in Ds],'err':[float(f'{Ds[k]-Dstar:.3e}') for k in range(5)]}

# ---- 14.5-1 flow chart
fig,ax=new(4.7,4.0)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.4): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.5,6.6,6.2,0.85,'Input $\\mathbf{X}$, $\\mathbf{y}$, weights $s_i$; check that every $y_i$ lies\nin the range of the deviance (Poisson $y\\geq 0$, gamma $y>0$)')
fb(3.5,5.55,6.2,0.6,'Start: $\\mathbf{w}=\\mathbf{0}$,  $b=\\ln\\bar y$  (so that $\\mu_i=\\bar y$)',fc='#E3F4EA',ec=GREEN)
ax.add_patch(Polygon([(3.5,5.0),(4.5,4.65),(3.5,4.3),(2.5,4.65)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.5,4.65,'solver',ha='center',va='center',fontsize=8.6,color=NAVY)
fb(1.55,3.0,2.9,1.65,'lbfgs (default)\ngradient, Eq. 14.3-10\nL-BFGS direction\nline search',fc='#F1EAF7',ec=PUR)
fb(5.45,3.0,2.9,1.65,'newton-cholesky\ngradient and Hessian\nsolve $\\mathbf{H}\\mathbf{s}=-\\nabla J$\nbacktracking search',fc='#F1EAF7',ec=PUR)
ax.plot([2.5,1.55],[4.65,4.65],color=NAVY,lw=0.9); A((1.55,4.65),(1.55,3.83)); ax.plot([4.5,5.45],[4.65,4.65],color=NAVY,lw=0.9); A((5.45,4.65),(5.45,3.83))
ax.add_patch(Polygon([(3.5,1.95),(4.7,1.55),(3.5,1.15),(2.3,1.55)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.5,1.55,'max $|g|\\leq$ tol?',ha='center',va='center',fontsize=8.2,color=NAVY)
ax.plot([1.55,1.55],[2.17,1.55],color=NAVY,lw=0.9); A((1.55,1.55),(2.3,1.55)); ax.plot([5.45,5.45],[2.17,1.55],color=NAVY,lw=0.9); A((5.45,1.55),(4.7,1.55))
fb(3.5,0.35,6.2,0.62,'Store coef_, intercept_, n_iter_;  warn if max_iter reached',fc=LGREY,ec=GREY)
A((3.5,1.15),(3.5,0.66)); ax.text(3.62,0.9,'yes',fontsize=8.2,color=NAVY)
ax.plot([4.7,7.45,7.45,3.5],[1.55,1.55,5.25,5.25],color=NAVY,lw=0.9,ls=(0,(4,2))); ax.text(6.1,1.4,'no: next iteration',fontsize=8.2,color=NAVY,va='top',ha='center')
A((3.5,6.17),(3.5,5.85)); A((3.5,5.25),(3.5,5.0))
ax.set_xlim(0,7.7); ax.set_ylim(0.0,7.1); save(fig,'f1451.png')

# ---- 14.7-1 alpha gallery
rng=np.random.default_rng(11); zg=np.sort(rng.uniform(-1.7,1.7,40)); yg=rng.poisson(np.exp(1.0+0.7*zg))
fig,axs=plt.subplots(1,4,figsize=(4.9,1.95),dpi=240,sharey=True); zz=np.linspace(-1.9,1.9,100); NUM['alpha_gallery']={}
for ax,a in zip(axs,[0,1,5,30]):
    m=PoissonRegressor(alpha=a,tol=1e-10,max_iter=1000).fit(zg[:,None],yg); NUM['alpha_gallery'][str(a)]={'w':round(float(m.coef_[0]),3),'b':round(float(m.intercept_),3)}
    pts(ax,zg,yg,s=12); ax.plot(zz,m.predict(zz[:,None]),color=GREEN,lw=1.8); ax.axhline(yg.mean(),color=GREY,lw=0.8,ls=':')
    ax.set_title(f'$\\alpha$ = {a}\n$w$ = {m.coef_[0]:.2f}',fontsize=8.8,color=PUR); ax.set_xlim(-1.9,1.9); ax.set_xticks([-1,0,1]); ax.grid(color=GRID,lw=0.5); ax.set_xlabel('$x$'); ax.set_ylim(-0.3,12)
axs[0].set_ylabel('$y$ and $\\mu$'); fig.tight_layout(w_pad=0.4); save(fig,'f1471.png')

# ---- 14.7-2 power on skewed data (left) and validation curve in alpha (right)
rng=np.random.default_rng(2); xp=rng.uniform(0,10,120); mp=1.0+0.35*xp**1.6; yp=rng.gamma(4,mp/4)   # curved mean, gamma noise
fig,axs=plt.subplots(1,2,figsize=(4.9,2.3),dpi=240); ax=axs[0]; xx=np.linspace(0,10,100)
pts(ax,xp,yp,s=10); NUM['power']={}
for p,c,ls,lab in [(0,GREY,':','$p$=0, identity link'),(1,BLUE,'-','$p$=1'),(2,GREEN,'--','$p$=2'),(3,PUR,'-.','$p$=3')]:
    m=TweedieRegressor(power=p,alpha=0,tol=1e-10,max_iter=1000).fit(xp[:,None],yp); ax.plot(xx,m.predict(xx[:,None]),color=c,lw=1.6,ls=ls,label=lab)
    NUM['power'][str(p)]={'pred10':round(float(m.predict([[10.]])[0]),2),'pred1':round(float(m.predict([[1.]])[0]),2)}
ax.legend(fontsize=7.6,loc='upper left',handlelength=2.2); ax_(ax,'$x$','$y$ and fitted $\\mu$'); ax.set_title('Tweedie power as the knob',fontsize=9,color=NAVY)
# validation curve
rng=np.random.default_rng(3); p_=30; wt=np.zeros(p_); wt[:5]=[0.4,-0.4,0.3,-0.3,0.35]
def mk(n): Xv=rng.normal(size=(n,p_)); return Xv,rng.poisson(np.exp(0.5+Xv@wt))
Xt,yt=mk(150); Xv,yv=mk(5000); al=np.logspace(-3,0.3,14); tr=[];va=[]
for a in al:
    m=PoissonRegressor(alpha=a,tol=1e-8,max_iter=1000).fit(Xt,yt); tr.append(mean_poisson_deviance(yt,m.predict(Xt))); va.append(mean_poisson_deviance(yv,m.predict(Xv)))
ib=int(np.argmin(va)); NUM['valcurve']={'best_alpha':round(float(al[ib]),4),'val_best':round(va[ib],4),'val_alpha0':round(va[0],4),'train_alpha0':round(tr[0],4),'val_at_largest':round(va[-1],4),'train_best':round(tr[ib],4)}
ax=axs[1]; ax.semilogx(al,tr,'-o',color=BLUE,ms=3,lw=1.4,label='training'); ax.semilogx(al,va,'-s',color=GREY,ms=3,lw=1.4,label='held-out')
ax.axvline(al[ib],color=PUR,lw=1,ls='--'); ax.text(al[ib]/1.2,1.25,f'best $\\alpha$ = {al[ib]:.2f}',fontsize=8.2,color=PUR,va='center',ha='right')
ax_(ax,'penalty $\\alpha$','mean Poisson deviance'); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(0.0,0.42)); ax.set_title('Validation curve ($n$=150, $p$=30)',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1472.png')

# ---- 14.9 case study
X_,expo,count,amount=make_portfolio(); freq=count/expo; tr_,te_=train_test_split(np.arange(len(X_)),test_size=0.25,random_state=0)
Xtr,Xte=X_.iloc[tr_],X_.iloc[te_]; wtr,wte=expo[tr_],expo[te_]
glm=PoissonRegressor(alpha=1e-4,tol=1e-8,max_iter=300).fit(Xtr,freq[tr_],sample_weight=wtr); ols=LinearRegression().fit(Xtr,freq[tr_],sample_weight=wtr)
mu_g=glm.predict(Xte); mu_o=np.clip(ols.predict(Xte),1e-3,None)
grp_age=np.where(Xte['young']>0,0,np.where(Xte['old']>0,2,1)); grp_pow=Xte['power_mid'].to_numpy()+2*Xte['power_high'].to_numpy()
labs=[];obs=[];pg_=[];po_=[];sz=[]
for a in range(3):
    for p in range(3):
        msk=(grp_age==a)&(grp_pow==p); e=wte[msk]; obs.append(np.sum(count[te_][msk])/e.sum()); pg_.append(np.sum(e*mu_g[msk])/e.sum()); po_.append(np.sum(e*mu_o[msk])/e.sum()); sz.append(msk.sum())
        labs.append(f'{["<30","30-60",">60"][a]}\n{["low","mid","high"][p]}')
NUM['groups']={'labels':[l.replace('\n',' ') for l in labs],'obs':[round(v,4) for v in obs],'glm':[round(v,4) for v in pg_],'ols':[round(v,4) for v in po_],'n':[int(v) for v in sz]}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.55),dpi=240,gridspec_kw={'width_ratios':[1.45,1]}); ax=axs[0]; i9=np.arange(9)
ax.bar(i9,obs,color='#C9D6EE',edgecolor=BLUE,lw=0.6,width=0.7,label='observed'); ax.plot(i9,pg_,'o',color=GREEN,ms=4.5,label='Poisson GLM',zorder=5); ax.plot(i9,po_,'x',color=OR,ms=5,mew=1.4,label='least squares',zorder=5)
ax.set_xticks(i9); ax.set_xticklabels(list('LMH')*3,fontsize=8); ax.tick_params(axis='x',length=0)
for c_,t_ in zip([1,4,7],['under 30','30 to 60','over 60']): ax.text(c_,-0.12,t_,fontsize=8,ha='center',va='top',transform=ax.get_xaxis_transform())
ax_(ax,'','claims per year'); ax.set_xlabel('age group; vehicle power L, M, H',labelpad=16); ax.legend(fontsize=7.6,loc='upper right')
ax.set_title('Frequency by group',fontsize=9,color=NAVY)
ax=axs[1]; names=list(TRUE.keys()); fit_=np.exp(glm.coef_); tru=np.exp(np.array(list(TRUE.values()))); yy=np.arange(5)[::-1]
ax.axvline(1,color=GREY,lw=0.7); ax.plot(tru,yy,'x',color=RED,ms=6,mew=1.5,label='true',zorder=6); ax.plot(fit_,yy,'o',color=GREEN,ms=5,label='fitted',zorder=5)
for a,b_,c in zip(fit_,tru,yy): ax.plot([a,b_],[c,c],color=GRID,lw=1,zorder=1)
ax.set_yticks(yy); ax.set_yticklabels(names,fontsize=8); ax_(ax,'factor $e^{w}$ on the mean',''); ax.set_title('Multiplicative effects',fontsize=9,color=NAVY); ax.legend(fontsize=7.6,loc='lower right'); ax.set_xlim(0.8,2.9)
fig.tight_layout(w_pad=0.8); save(fig,'f1491.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

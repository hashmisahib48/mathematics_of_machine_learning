# All figures of Chapter 13. Run: python3 fig/figs.py   (PNGs into ch13/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.filterwarnings('ignore')
from figstyle import *
from itertools import combinations
from sklearn.linear_model import *
from sklearn.linear_model._ransac import _dynamic_max_trials
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
NUM={}
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,20.]); X1=x[:,None]
def axes(ax,xl='distance $x$ (km)',yl='time $y$ (min)'):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx=x,yy=y,s=34,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,**k)
FS=9.5

# ---- 13.0-1 opener
rng=np.random.default_rng(4); xo=rng.uniform(0.3,9.7,34); yo=1.5+0.8*xo+rng.normal(0,0.55,34)
bad=np.argsort(np.abs(xo-7))[:5]; yo[bad]+=rng.uniform(5.5,8,5)
mo=LinearRegression().fit(xo[:,None],yo); mh=HuberRegressor().fit(xo[:,None],yo); xx=np.array([0,10.])
fig,ax=plt.subplots(figsize=(5.6,1.75),dpi=240)
ok=np.ones(34,bool); ok[bad]=False; pts(ax,xo[ok],yo[ok],s=24); ax.scatter(xo[bad],yo[bad],s=34,color=BLUE,edgecolor=RED,lw=1.4,zorder=5)
ax.plot(xx,mo.predict(xx[:,None]),color=GREY,lw=1.8,ls='--',zorder=3); ax.plot(xx,mh.predict(xx[:,None]),color=GREEN,lw=2.2,zorder=4)
ax.text(9.9,2.0,'robust fit',color=GREEN,ha='right',fontsize=12,weight='bold')
ax.text(0.2,mo.predict([[9.5]])[0]+0.6,'least squares,\npulled upwards',color=GREY,ha='left',va='bottom',fontsize=11.5,transform=ax.transData)
ax.annotate('outliers',xy=(xo[bad][0],yo[bad][0]+0.3),xytext=(3.4,13.2),color=RED,fontsize=12,arrowprops=dict(arrowstyle='-',color=RED,lw=0.7))
ax.set_xlim(0,10); ax.set_ylim(0,15); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('feature  $x$',fontsize=11.5); ax.set_ylabel('target  $y$',fontsize=11.5); save(fig,'f1300.png')

# ---- 13.1-1 the problem
ols=LinearRegression().fit(X1,y); NUM['ols']={'w':round(float(ols.coef_[0]),2),'b':round(float(ols.intercept_),2),'p6':round(float(ols.predict([[6]])[0]),2),'r':[round(float(v),2) for v in y-ols.predict(X1)]}
fig,ax=plt.subplots(figsize=(4.4,2.6),dpi=240); xx=np.array([0,6.6])
ax.plot(xx,0.9+1.7*xx,color=GREEN,lw=1.8); ax.plot(xx,ols.intercept_+ols.coef_[0]*xx,color=GREY,lw=1.8,ls='--')
yb=np.array([3,4,6,7,10.]); ax.scatter(x[:4],y[:4],s=40,color=BLUE,edgecolor='white',zorder=5); ax.scatter([5],[20],s=44,color=BLUE,edgecolor=RED,lw=1.5,zorder=5); ax.plot(5,10,'o',mfc='white',mec=GREEN,ms=6,zorder=4)
ax.plot([5,5],[10,20],color=RED,lw=1,ls=':'); ax.annotate('wrong entry (20, not 10)',xy=(4.92,20),xytext=(1.9,20.3),color=RED,fontsize=FS,va='center',arrowprops=dict(arrowstyle='-',color=RED,lw=0.6))
ax.axvline(6,color=PUR,lw=0.9,ls='--'); ax.plot(6,11.1,'o',color=GREEN,ms=5,zorder=6); ax.plot(6,19.1,'s',color=GREY,ms=5,zorder=6)
ax.text(6.2,11.1,'11.1',color=GREEN,fontsize=FS,ha='left',va='center'); ax.text(6.2,19.1,'19.1',color=GREY,fontsize=FS,ha='left',va='center')
ax.text(0.15,16.4,'$\\hat y = -3.1 + 3.7x$  (all five)',color=GREY,fontsize=FS); ax.text(0.15,14.0,'$\\hat y = 0.9 + 1.7x$  (clean data)',color=GREEN,fontsize=FS)
axes(ax); ax.set_xlim(0,6.9); ax.set_ylim(-4,22); save(fig,'f1311.png')

# ---- 13.2-1 losses and influence
z=np.linspace(-3,3,601); eps=1.35; tau=0.8
sq=z**2; ab=np.abs(z); hub=np.where(np.abs(z)<=eps,z**2,2*eps*np.abs(z)-eps**2); pin=np.where(z>=0,tau*z,(tau-1)*z)
psi_sq=2*z; psi_ab=np.sign(z); psi_h=np.where(np.abs(z)<=eps,2*z,2*eps*np.sign(z)); psi_p=np.where(z>=0,tau,tau-1.)
fig,axs=plt.subplots(1,2,figsize=(5.8,2.35),dpi=240)
ax=axs[0]; ax.plot(z,sq,color=RED,lw=1.6,label='squared  $r^2$'); ax.plot(z,ab,color=OR,lw=1.6,ls='--',label='absolute  $|r|$'); ax.plot(z,hub,color=GREEN,lw=2.0,label='Huber  $H_\\epsilon$'); ax.plot(z,pin,color=PUR,lw=1.6,ls=':',label='pinball  $\\rho_{0.8}$')
ax.set_ylim(0,6); ax.set_xlim(-3,3); ax.legend(fontsize=9.2,loc='upper center',handlelength=2.0,borderpad=0.2,labelspacing=0.25); ax.set_xlabel('residual $r$ (in units of $\\sigma$)'); ax.set_ylabel('penalty'); ax.set_title('Loss',fontsize=FS+0.5,color=NAVY); ax.grid(color=GRID,lw=0.5)
ax=axs[1]; ax.plot(z,psi_sq,color=RED,lw=1.6); ax.plot(z,psi_ab,color=OR,lw=1.6,ls='--'); ax.plot(z,psi_h,color=GREEN,lw=2.0); ax.plot(z,psi_p,color=PUR,lw=1.6,ls=':')
ax.set_ylim(-3.6,3.6); ax.set_xlim(-3,3); ax.set_xlabel('residual $r$ (in units of $\\sigma$)'); ax.set_ylabel('pull on the line'); ax.set_title('Slope of the loss',fontsize=FS+0.5,color=NAVY); ax.grid(color=GRID,lw=0.5)
ax.text(0.25,-2.7,'squared: no limit',color=RED,fontsize=9.2,ha='left',va='center')
fig.tight_layout(w_pad=1.2); save(fig,'f1321.png')

# ---- 13.3 Huber fixed point numbers and weights picture
h=HuberRegressor(epsilon=1.35,alpha=0).fit(X1,y); b,w=h.intercept_,h.coef_[0]; r=y-b-w*x; sg=h.scale_; om=np.minimum(1,1.35*sg/np.abs(r))
NUM['huber']={'b':round(float(b),4),'w':round(float(w),4),'sigma':round(float(sg),4),'r':[round(float(v),4) for v in r],'z':[round(float(v),3) for v in np.abs(r)/sg],'esig':round(float(1.35*sg),4),'omega':[round(float(v),4) for v in om],
 'sumsq_in':round(float((r[:3]**2).sum()),4),'den':round(float(5-2*1.35**2),4),'sig2':round(float((r[:3]**2).sum()/(5-2*1.35**2)),4)}
S=[om.sum(),(om*x).sum(),(om*x*x).sum(),(om*y).sum(),(om*x*y).sum()]; NUM['huber']['S']=[round(float(v),4) for v in S]
Ar=np.array([[S[0],S[1]],[S[1],S[2]]]); NUM['huber']['det']=round(float(np.linalg.det(Ar)),4); NUM['huber']['solve']=[round(float(v),4) for v in np.linalg.solve(Ar,S[3:])]
Sr=[round(v,4) for v in S]; Ar2=np.array([[Sr[0],Sr[1]],[Sr[1],Sr[2]]]); NUM['huber']['solve_rounded']=[round(float(v),4) for v in np.linalg.solve(Ar2,Sr[3:])]
NUM['huber']['p6']=round(float(h.predict([[6]])[0]),2)
S0,S1,S2,Sy,Sxy=3.7962,9.268,27.4889,19.6568,57.2942   # sums as printed by code/toy_huber.py
dt=S0*S2-S1**2; NUM['huber_cramer']={'det':round(dt,4),'num_b':round(S2*Sy-S1*Sxy,4),'num_w':round(S0*Sxy-S1*Sy,4),'b':round((S2*Sy-S1*Sxy)/dt,4),'w':round((S0*Sxy-S1*Sy)/dt,4)}
NUM['huber_tab']=[[int(i+1),int(x[i]),int(y[i]),round(float(r[i]),4),round(float(abs(r[i])/sg),3),round(float(om[i]),4),round(float(om[i]*x[i]),4),round(float(om[i]*y[i]),4),round(float(om[i]*x[i]*x[i]),4),round(float(om[i]*x[i]*y[i]),4)] for i in range(5)]
# Huber alpha=1e-4 default and eps sweep on toy
NUM['huber_eps']={str(e):[round(float(HuberRegressor(epsilon=e,alpha=0,max_iter=2000,tol=1e-10).fit(X1,y).coef_[0]),3),round(float(HuberRegressor(epsilon=e,alpha=0,max_iter=2000,tol=1e-10).fit(X1,y).intercept_),3)] for e in (1.2,1.35,1.4,1.5)}
fig,ax=plt.subplots(figsize=(3.9,2.3),dpi=240); zz=np.logspace(-0.5,1.4,400); ax.plot(zz,np.minimum(1,1.35/np.maximum(zz,1e-9)),color=GREEN,lw=1.8); ax.set_xscale('log')
zi=np.abs(r)/sg
for k,(zv,ov) in enumerate(zip(zi,om)):
    ax.plot(zv,ov,'o',color=BLUE,ms=5.5,mec='white',zorder=5,clip_on=False); ax.text(zv,ov+0.07,f'{k+1}',fontsize=FS,color=NAVY,ha='center')
ax.axvline(1.35,color=PUR,lw=0.9,ls='--'); ax.text(1.5,0.5,'$\\epsilon = 1.35$',color=PUR,fontsize=FS)
ax.set_xlabel('$|r_i|\\,/\\,\\sigma$'); ax.set_ylabel('weight $\\omega_i$'); ax.set_ylim(0,1.15); ax.set_xlim(0.3,25); ax.set_xticks([0.5,1,2,5,10,20]); ax.set_xticklabels(['0.5','1','2','5','10','20']); ax.grid(color=GRID,lw=0.5); save(fig,'f1331.png')

# ---- 13.4 ten lines and slopes
pairs=list(combinations(range(5),2)); lines=[]; t=2.0
for i,j in pairs:
    ww=(y[j]-y[i])/(x[j]-x[i]); bb=y[i]-ww*x[i]; rr=y-bb-ww*x; lines.append((i,j,bb,ww,np.abs(rr).sum(),int((np.abs(rr)<=t).sum())))
NUM['pairs']=[[i+1,j+1,round(float(bb),3),round(float(ww),3),round(float(s),2),n] for i,j,bb,ww,s,n in lines]
sl=np.sort([l[3] for l in lines]); NUM['slopes_sorted']=[round(float(v),3) for v in sl]; NUM['ts_classic']=[float(np.median(sl)),float(np.median(y-np.median(sl)*x))]
tsr=TheilSenRegressor(random_state=0).fit(X1,y); NUM['ts_sk']=[round(float(tsr.coef_[0]),3),round(float(tsr.intercept_),3),round(float(tsr.breakdown_),3)]
cl=LinearRegression().fit(X1,np.array([3,4,6,7,10.])); NUM['clean_slopes_median']=float(np.median([(yb[j]-yb[i])/(x[j]-x[i]) for i,j in pairs]))
fig,axs=plt.subplots(1,2,figsize=(5.8,2.35),dpi=240,gridspec_kw={'width_ratios':[1.15,1]}); ax=axs[0]; xx=np.array([0,5.6])
for i,j,bb,ww,s,n in lines:
    if n==4: ax.plot(xx,bb+ww*xx,color=GREEN,lw=1.1,alpha=0.85,zorder=2)
    else: ax.plot(xx,bb+ww*xx,color=GREY,lw=0.9,ls='--',alpha=0.8,zorder=1)
ax.fill_between(xx,1.5+1.5*xx-2,1.5+1.5*xx+2,color=LB,alpha=0.8,zorder=0); pts(ax,x[:4],y[:4]); ax.scatter([5],[20],s=34,color=BLUE,edgecolor=RED,lw=1.4,zorder=5)
ax.text(0.15,17.5,'6 lines with 4 inliers',color=GREEN,fontsize=9.2); ax.text(0.15,15.4,'4 lines with 2 inliers',color=GREY,fontsize=9.2); ax.text(0.15,12.8,'blue band: 2 either\nside of the line\nthrough points 1, 3',color=BLUE,fontsize=9.2,ha='left',va='top')
ax.set_xlim(0,5.6); ax.set_ylim(-2,22); axes(ax,'$x$','$y$'); ax.set_title('Ten lines through two points',fontsize=FS,color=NAVY)
ax=axs[1]; ax.axhline(0,color=GRID,lw=0.5)
for k,v in enumerate(sl): ax.plot(v,0.25*(k%2)-0.12,'o',color=BLUE,ms=5,mec='white',zorder=4)
ax.set_xscale('log'); ax.plot([1.75,1.75],[-0.45,0.45],color=GREEN,lw=2); ax.text(1.75,0.58,'median 1.75',color=GREEN,ha='center',fontsize=9.2)
ax.plot([3.7,3.7],[-0.45,0.45],color=GREY,lw=1.6,ls='--'); ax.text(3.7,-0.78,'least squares 3.7',color=GREY,ha='center',fontsize=9.2,va='top')
ax.set_ylim(-1.2,1.0); ax.set_yticks([]); ax.set_xlim(0.8,17); ax.set_xticks([1,2,4,8,16]); ax.set_xticklabels(['1','2','4','8','16']); ax.set_xlabel('slope of a pair (min/km)'); ax.set_title('Ten pairwise slopes',fontsize=FS,color=NAVY); ax.spines['left'].set_visible(False)
fig.tight_layout(w_pad=1.0); save(fig,'f1341.png')

# ---- numbers: trials table
tab={}
for f_ in (0.9,0.8,0.7,0.5):
    tab[str(f_)]={}
    for s_ in (2,3,5,11):
        N=int(np.ceil(np.log(0.01)/np.log(1-f_**s_))); assert N==int(_dynamic_max_trials(f_*1000,1000,s_,0.99)); tab[str(f_)][str(s_)]=N
NUM['trials']=tab
NUM['breakdown']={str(s_):round(1-0.5**(1/s_),4) for s_ in (2,3,5,11)}
NUM['breakdown_sk']={str((n_,s_)):round(float(1-(0.5**(1/s_)*(n_-s_+1)+s_-1)/n_),3) for n_,s_ in ((5,2),(100,2),(1000,2),(331,11))}

# ---- 13.5 pinball and best constant
wt=np.array([2.,3,4,5,16]); cs=np.linspace(0,18,721)
def pj(c,t): r_=wt[None,:]-np.atleast_1d(c)[:,None]; return np.where(r_>=0,t*r_,(t-1)*r_).sum(1)
NUM['pin_table']=[[c_,round(float(pj(c_,0.5)[0]),2),round(float(pj(c_,0.9)[0]),2)] for c_ in (2,3,4,5,6,16)]
fig,axs=plt.subplots(1,2,figsize=(5.8,2.3),dpi=240); ax=axs[0]; rr=np.linspace(-3,3,300)
for t_,c_,ls_ in ((0.1,BLUE,':'),(0.5,GREEN,'-'),(0.9,PUR,'--')): ax.plot(rr,np.where(rr>=0,t_*rr,(t_-1)*rr),color=c_,lw=1.7,ls=ls_,label=f'$\\tau = {t_}$')
ax.legend(fontsize=9.2,loc='upper center',handlelength=2.0,borderpad=0.2); ax.set_xlabel('residual $r = y - \\hat y$'); ax.set_ylabel('$\\rho_\\tau(r)$'); ax.set_title('Pinball loss',fontsize=FS+0.5,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_ylim(0,2.9)
ax=axs[1]
for t_,c_,ls_ in ((0.5,GREEN,'-'),(0.9,PUR,'--')):
    ax.plot(cs,pj(cs,t_),color=c_,lw=1.7,ls=ls_); cb=cs[np.argmin(pj(cs,t_))]; ax.plot(cb,pj(cb,t_)[0],'o',color=c_,ms=6,mec='white',zorder=5)
ax.plot(wt,np.full(5,-2.6),'|',color=BLUE,ms=9,mew=1.6); ax.text(6.6,-2.6,'the five waits',color=BLUE,fontsize=9.2,ha='left',va='center')
ax.text(6.6,3.2,'best $c$ = 4  ($\\tau$ = 0.5)',color=GREEN,fontsize=9.2,ha='left',va='center'); ax.text(13.0,30,'best $c$ = 16\n($\\tau$ = 0.9)',color=PUR,fontsize=9.2,ha='right',va='top')
ax.set_xlabel('constant $c$'); ax.set_ylabel('total penalty'); ax.set_title('Best constant for 2, 3, 4, 5, 16',fontsize=FS+0.5,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_ylim(-4.2,35); ax.set_xlim(0,18)
fig.tight_layout(w_pad=1.0); save(fig,'f1351.png')

# ---- 13.7 RANSAC flow chart
fig,ax=new(5.8,3.3)
def fb(x0,y0,w_,h_,t_,fc=LB,ec=BLUE,fs=9.2): box(ax,x0,y0,w_,h_,t_,fc=fc,ec=ec,fs=fs,r=0.07)
A=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
def dia(cx,cy,w_,h_,t_): ax.add_patch(Polygon([(cx,cy+h_/2),(cx+w_/2,cy),(cx,cy-h_/2),(cx-w_/2,cy)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(cx,cy,t_,ha='center',va='center',fontsize=9.2,color=NAVY)
fb(0.1,3.85,2.9,1.0,'Draw $s$ points\nat random')
fb(3.55,3.85,2.9,1.0,'Fit the base model\nto those $s$ points')
fb(7.0,3.85,2.9,1.0,'Inliers: $|r_i| \\leq t$\namong all $n$ points')
dia(8.45,2.45,3.1,1.5,'more inliers\nthan the best?'); fb(3.7,1.95,2.5,1.0,'Keep as best;\nupdate $N$ (13.4-2)',fc='#E3F4EA',ec=GREEN); dia(1.55,2.45,2.9,1.5,'trials used up\nor $N$ reached?')
fb(0.1,0.3,6.1,0.8,'Refit on the inliers of the best set;\nstore estimator_, inlier_mask_',fc='#E3F4EA',ec=GREEN)
A((3.0,4.35),(3.55,4.35)); A((6.45,4.35),(7.0,4.35)); A((8.45,3.85),(8.45,3.2)); A((6.9,2.45),(6.2,2.45)); A((3.7,2.45),(3.0,2.45)); A((1.55,1.7),(1.55,1.1)); A((1.55,3.2),(1.55,3.85))
ax.plot([8.45,8.45,3.35,3.35],[1.7,1.45,1.45,2.45],color=NAVY,lw=0.9)
ax.text(6.55,2.58,'yes',fontsize=9.2,color=GREEN,ha='center',va='bottom'); ax.text(8.6,1.55,'no',fontsize=9.2,color=RED,va='bottom'); ax.text(1.7,1.4,'yes',fontsize=9.2,color=GREEN,va='center'); ax.text(1.7,3.5,'no',fontsize=9.2,color=RED,va='center')
ax.set_xlim(0,10); ax.set_ylim(0.1,5.0); save(fig,'f1371.png')

# ---- 13.10 gallery: Huber epsilon (top) and RANSAC threshold (bottom) on the delivery data
fig,axs=plt.subplots(2,4,figsize=(5.8,3.0),dpi=240,sharex=True,sharey=True); xx=np.array([0,5.6]); NUM['gallery']={'huber':{},'ransac':{}}
for ax,e in zip(axs[0],(1.2,1.35,1.4,1.5)):
    m=HuberRegressor(epsilon=e,alpha=0,max_iter=2000,tol=1e-10).fit(X1,y); ax.plot(xx,m.intercept_+m.coef_[0]*xx,color=GREEN,lw=1.8)
    NUM['gallery']['huber'][str(e)]=[round(float(m.coef_[0]),2),round(float(m.intercept_),2)]
    pts(ax,s=20); ax.set_title(f'$\\epsilon$ = {e:g}',fontsize=FS,color=PUR); ax.text(0.2,17,f'$w$ = {m.coef_[0]:.2f}',fontsize=9.2,color=GREEN)
for ax,t_ in zip(axs[1],(0.3,1,2,12)):
    m=RANSACRegressor(residual_threshold=t_,random_state=0,max_trials=100).fit(X1,y); e_=m.estimator_; ax.plot(xx,e_.intercept_+e_.coef_[0]*xx,color=GREEN,lw=1.8)
    NUM['gallery']['ransac'][str(t_)]=[round(float(e_.coef_[0]),2),round(float(e_.intercept_),2),int(m.inlier_mask_.sum())]
    pts(ax,s=20); ax.scatter(x[~m.inlier_mask_],y[~m.inlier_mask_],s=36,facecolor='none',edgecolor=RED,lw=1.2,zorder=6); ax.set_title(f'$t$ = {t_:g}',fontsize=FS,color=PUR); ax.text(0.2,17,f'$w$ = {e_.coef_[0]:.2f}',fontsize=9.2,color=GREEN)
for ax in axs.ravel(): ax.grid(color=GRID,lw=0.5); ax.set_xlim(0,5.6); ax.set_ylim(-4,22); ax.tick_params(labelsize=9.2)
for ax in axs[1]: ax.set_xlabel('$x$')
axs[0,0].set_ylabel('Huber\n$y$'); axs[1,0].set_ylabel('RANSAC\n$y$'); fig.tight_layout(w_pad=0.4,h_pad=0.8); save(fig,'f13101.png')

# ---- 13.10 validation curve for epsilon on contaminated data
rng=np.random.default_rng(3); Xv,yv,cv_=make_regression(n_samples=300,n_features=5,n_informative=5,noise=8,coef=True,random_state=1)
Xt,yt=Xv[200:],yv[200:]; Xv,yv=Xv[:200],yv[:200].copy(); badv=rng.choice(200,24,replace=False); yv[badv]+=rng.choice([-1,1],24)*rng.uniform(150,300,24)
eps_=np.round(np.logspace(0,np.log10(8),15),3); tr_,va_=validation_curve(HuberRegressor(alpha=0,max_iter=2000),Xv,yv,param_name='epsilon',param_range=eps_,cv=5,scoring='neg_median_absolute_error')
cvmae=-va_.mean(1); tst=[mean_absolute_error(yt,HuberRegressor(epsilon=e,alpha=0,max_iter=2000).fit(Xv,yv).predict(Xt)) for e in eps_]
olsm=mean_absolute_error(yt,LinearRegression().fit(Xv,yv).predict(Xt)); NUM['valcurve']={'eps':[float(e) for e in eps_],'cv_mae':[round(float(v),2) for v in cvmae],'test_mae':[round(float(v),2) for v in tst],'ols_test_mae':round(float(olsm),2),'best_eps':float(eps_[int(np.argmin(cvmae))])}
fig,ax=plt.subplots(figsize=(4.4,2.3),dpi=240); ax.plot(eps_,cvmae,'-o',color=BLUE,ms=3.5,lw=1.5,label='5-fold CV (median error)'); ax.plot(eps_,tst,'-s',color=GREY,ms=3.5,lw=1.5,ls='--',label='clean test data (mean error)')
ax.axhline(olsm,color=RED,lw=1,ls=':'); ax.text(1.02,olsm-0.6,'least squares (test)',color=RED,fontsize=9.2,va='top'); ax.axvline(1.35,color=PUR,lw=0.9,ls='--'); ax.text(1.42,ax.get_ylim()[0]+1,'default',color=PUR,fontsize=9.2)
ax.set_xscale('log'); ax.set_xticks([1,1.35,2,3,5,8]); ax.set_xticklabels(['1','1.35','2','3','5','8']); ax.minorticks_off(); ax.set_xlabel('$\\epsilon$ (epsilon)'); ax.set_ylabel('absolute error'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=9,loc='upper left',borderpad=0.2,labelspacing=0.25); ax.set_ylim(top=olsm*1.45)
save(fig,'f13102.png')

# ---- 13.10 comparison on contaminated data + breakdown curve
def contam(rng,n,m):
    xx_=rng.uniform(0,10,n); yy_=1+0.8*xx_+rng.normal(0,0.6,n); idx=rng.choice(n,m,replace=False); xx_[idx]=rng.uniform(7,10,m); yy_[idx]=rng.uniform(-3,2,m); return xx_[:,None],yy_,idx
def mods(): return {'OLS':LinearRegression(),'Huber':HuberRegressor(max_iter=500),'RANSAC':RANSACRegressor(residual_threshold=1.0,random_state=0),'Theil-Sen':TheilSenRegressor(random_state=0),'Quantile':QuantileRegressor(alpha=0)}
sty={'OLS':(GREY,'--',1.6),'Huber':(GREEN,'-',1.8),'RANSAC':(OR,'-.',1.8),'Theil-Sen':(PUR,':',2.2),'Quantile':(BLUE,(0,(5,1.5)),1.4)}
fig,axs=plt.subplots(1,2,figsize=(5.8,2.5),dpi=240,gridspec_kw={'width_ratios':[1,1.05]}); ax=axs[0]; X_,y_,idx=contam(np.random.default_rng(5),40,8); NUM['compare']={}
ok=np.ones(40,bool); ok[idx]=False; ax.scatter(X_[ok,0],y_[ok],s=16,color=BLUE,edgecolor='white',lw=0.5,zorder=5); ax.scatter(X_[idx,0],y_[idx],s=26,color=BLUE,edgecolor=RED,lw=1.2,zorder=5)
for k,m in mods().items():
    m.fit(X_,y_); e_=getattr(m,'estimator_',m); NUM['compare'][k]=[round(float(e_.coef_[0]),3),round(float(e_.intercept_),3)]; c_,ls_,lw_=sty[k]; ax.plot([0,10],[e_.intercept_,e_.intercept_+10*e_.coef_[0]],color=c_,ls=ls_,lw=lw_,label=k)
ax.legend(fontsize=9,loc='upper left',borderpad=0.2,labelspacing=0.2,handlelength=2.4); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.grid(color=GRID,lw=0.5); ax.set_title('40 points, 8 bad ones',fontsize=FS,color=NAVY); ax.set_ylim(-3.5,14)
ax=axs[1]; ms_=list(range(0,21,2)); res={k:[] for k in mods()}
for m_ in ms_:
    er={k:[] for k in mods()}
    for sd in range(30):
        X_,y_,_=contam(np.random.default_rng(100+sd),40,m_)
        for k,mod in mods().items(): mod.fit(X_,y_); e_=getattr(mod,'estimator_',mod); er[k].append(abs(e_.coef_[0]-0.8))
    for k in res: res[k].append(float(np.median(er[k])))
for k in res: c_,ls_,lw_=sty[k]; ax.plot([100*m/40 for m in ms_],res[k],color=c_,ls=ls_,lw=lw_)
NUM['breakdown_curve']={'pct':[100*m/40 for m in ms_],**{k:[round(v,3) for v in vv] for k,vv in res.items()}}
ax.set_xlabel('share of bad points (%)'); ax.set_ylabel('error of the slope (median of 30)'); ax.grid(color=GRID,lw=0.5); ax.set_title('Slope error as bad points grow',fontsize=FS,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f13103.png')

# ---- 13.10 quantile band (same generator and seed as quantile_interval.py)
rng=np.random.default_rng(0)
def mk(n):
    x_=rng.uniform(0,10,n); y_=2+1.5*x_+rng.normal(size=n)*(0.3+0.35*x_); return x_[:,None],y_
Xq,yq=mk(400); Xqt,yqt=mk(4000); fig,ax=plt.subplots(figsize=(4.6,2.5),dpi=240); NUM['quant']={}
ax.scatter(Xq[:,0],yq,s=9,color=BLUE,alpha=0.7,lw=0,zorder=3); xg=np.linspace(0,10,50)[:,None]; fits={}
for t_,c_,ls_ in ((0.05,PUR,'--'),(0.5,GREEN,'-'),(0.95,PUR,'--')):
    m=QuantileRegressor(quantile=t_,alpha=0).fit(Xq,yq); fits[t_]=m.predict(xg); ax.plot(xg[:,0],fits[t_],color=c_,ls=ls_,lw=1.8)
    NUM['quant'][str(t_)]={'b':round(float(m.intercept_),3),'w':round(float(m.coef_[0]),3),'below':round(float(np.mean(yqt<=m.predict(Xqt))),3)}
ax.fill_between(xg[:,0],fits[0.05],fits[0.95],color=LB,alpha=0.7,zorder=1); ax.text(10.15,fits[0.95][-1],'$\\tau$ = 0.95',color=PUR,fontsize=9.2,ha='left',va='center'); ax.text(10.15,fits[0.05][-1],'$\\tau$ = 0.05',color=PUR,fontsize=9.2,ha='left',va='center'); ax.text(10.15,fits[0.5][-1],'$\\tau$ = 0.5',color=GREEN,fontsize=9.2,ha='left',va='center')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.grid(color=GRID,lw=0.5); ax.set_xlim(0,12.2); ax.set_xticks([0,2,4,6,8,10]); save(fig,'f13104.png')

# ---- 13.12 case study figure
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0)
rng=np.random.default_rng(0); badc=rng.choice(len(ytr),size=33,replace=False); ybad=ytr.copy(); ybad[badc]+=300
cm={'OLS':LinearRegression(),'Huber':HuberRegressor(max_iter=1000),'RANSAC':RANSACRegressor(min_samples=60,residual_threshold=100,max_trials=200,random_state=0),'Theil-Sen':TheilSenRegressor(random_state=0),'Quantile':QuantileRegressor(alpha=0.0)}
NUM['case']={}
for k,m in cm.items(): m.fit(Xtr,ybad); p=m.predict(Xte); NUM['case'][k]=[round(float(r2_score(yte,p)),3),round(float(mean_absolute_error(yte,p)),1)]
fig,axs=plt.subplots(1,2,figsize=(5.8,2.45),dpi=240,gridspec_kw={'width_ratios':[1,1.15]}); ax=axs[0]; names=list(cm); vals=[NUM['case'][k][1] for k in names]
ref=mean_absolute_error(yte,LinearRegression().fit(Xtr,ytr).predict(Xte)); ax.barh(names[::-1],vals[::-1],color=[GREEN if v<50 else (GREY if k=='OLS' else OR) for k,v in zip(names[::-1],vals[::-1])],height=0.6)
ax.axvline(ref,color=BLUE,lw=1.2,ls='--'); ax.text(ref-0.5,4.75,'OLS on clean data',color=BLUE,fontsize=9.2,ha='right',va='center'); ax.set_ylim(-0.5,5.1)
for i_,v in enumerate(vals[::-1]): ax.text(v+0.5,i_,f'{v:.1f}',fontsize=9.2,va='center')
ax.set_xlim(30,62); ax.set_xlabel('test MAE (smaller is better)'); ax.set_title('Trained on wrecked targets',fontsize=FS,color=NAVY); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax=axs[1]; hm=cm['Huber']; res_=ybad-hm.predict(Xtr); good=np.ones(len(ytr),bool); good[badc]=False
ax.scatter(np.where(good)[0],res_[good],s=7,color=BLUE,lw=0); ax.scatter(np.where(~good)[0],res_[~good],s=12,color=RED,lw=0); ax.text(165,170,'red: the 33 wrecked targets',color=RED,fontsize=9.2,ha='center')
ax.axhspan(-1.35*hm.scale_,1.35*hm.scale_,color=LB,alpha=0.8,zorder=0); ax.text(5,1.35*hm.scale_*0.35,'$|r| \\leq \\epsilon\\sigma$',color=BLUE,fontsize=9.2,va='center')
ax.set_xlabel('training sample number'); ax.set_ylabel('Huber residual'); ax.set_title('Residuals of the Huber fit',fontsize=FS,color=NAVY); ax.grid(color=GRID,lw=0.5)
fig.tight_layout(w_pad=1.0); save(fig,'f13121.png')
NUM['case_clean']={}
for k,mm in {'OLS':LinearRegression(),'Huber':HuberRegressor(max_iter=1000),'Theil-Sen':TheilSenRegressor(random_state=0),'Quantile':QuantileRegressor(alpha=0.0)}.items():
    mm.fit(Xtr,ytr); p=mm.predict(Xte); NUM['case_clean'][k]=[round(float(r2_score(yte,p)),3),round(float(mean_absolute_error(yte,p)),1)]
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)
print(json.dumps({k:NUM[k] for k in ('gallery','valcurve','compare','quant','case')},indent=0)[:3000])

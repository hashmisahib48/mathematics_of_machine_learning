# All figures of Chapter 12. Run: python3 fig/figs.py  (PNGs go into ch12/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Polygon
from sklearn.linear_model import BayesianRidge, ARDRegression, Ridge, LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from bayes_scratch import MyBayesRidge
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9.5})
NUM={}
GL='#CFE8DA'          # light green for bands
x=np.array([-3.,-1,1,3]); y=np.array([-2.,-2,1,3]); A0,L0=1.0,5.0
Sxx=float(x@x); prec=L0+A0*Sxx; M0=A0*float(x@y)/prec; V0=1/prec
def gx(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx=x,yy=y,s=40,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=6,**k)

# ---- 12.0-1 opener: data, mean, band, posterior samples
rng=np.random.default_rng(4); xo=np.sort(rng.uniform(2.5,7.0,14)); yo=1.0+0.8*xo+rng.normal(0,0.55,14)
Xo=xo[:,None]; br=BayesianRidge().fit(Xo,yo); xg=np.linspace(0,10,200); mu,sd=br.predict(xg[:,None],return_std=True)
fig,ax=plt.subplots(figsize=(5.2,1.85),dpi=240)
ax.fill_between(xg,mu-2*sd,mu+2*sd,color=GL,lw=0,zorder=1)
sl=rng.normal(br.coef_[0],np.sqrt(br.sigma_[0,0]),12)
for s_ in sl: ax.plot(xg,yo.mean()+s_*(xg-xo.mean()),color=GREEN,lw=0.7,alpha=0.55,zorder=2)
ax.plot(xg,mu,color=GREEN,lw=2.2,zorder=3); pts(ax,xo,yo,s=30)
ax.text(5.0,2.2,'data',color=BLUE,fontsize=10.5,ha='left')
ax.annotate('band: mean $\\pm$ 2 s.d.',xy=(8.4,mu[168]-2*sd[168]),xytext=(9.8,1.3),fontsize=10.5,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6),ha='right',va='center')
ax.annotate('thin lines: posterior samples',xy=(1.2,yo.mean()+(sl[0])*(1.2-xo.mean())),xytext=(0.3,8.6),fontsize=10.5,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6),ha='left',va='center')
ax.set_xlim(0,10); ax.set_ylim(-0.5,12.4); ax.set_xlabel('feature  $x$'); ax.set_ylabel('target  $y$'); ax.set_xticks([]); ax.set_yticks([]); save(fig,'f1200.png')
NUM['opener']={'alpha_':float(br.alpha_),'lambda_':float(br.lambda_)}

# ---- 12.1-1 the calibration data
fig,ax=plt.subplots(figsize=(4.2,2.4),dpi=240); pts(ax,s=50); gx(ax)
for a,b_ in zip(x,y): ax.text(a+0.17,b_-0.62,f'({a:.0f}, {b_:.0f})'.replace('-','\u2212'),fontsize=8.5,color=GREY)
ax.axvline(6,color=PUR,lw=1,ls='--'); ax.text(5.85,-2.9,'$x_*$ = 6:\nhow sure?',color=PUR,fontsize=9,ha='right',va='bottom')
ax.set_xlim(-4.5,7.2); ax.set_ylim(-4,6.2); ax.set_xlabel('setting  $x$ (centred)'); ax.set_ylabel('reading  $y$ (centred)'); save(fig,'f1211.png')

# ---- 12.2-1 prior, posterior, predictive band
rng=np.random.default_rng(2); xx=np.array([-7,7.]); xs=np.linspace(-7,7,200)
fig,axs=plt.subplots(1,3,figsize=(5.4,2.15),dpi=240,sharey=True)
for ax,t in zip(axs,['Prior: slopes from $\\mathcal{N}(0,1/\\lambda)$','Posterior: slopes left','Predictive band, $\\pm$ 2 s.d.']):
    ax.set_title(t,fontsize=8.8,color=NAVY); gx(ax); ax.set_xlim(-7,7); ax.set_ylim(-9,9); ax.set_xlabel('$x$')
for s_ in rng.normal(0,L0**-0.5,14): axs[0].plot(xx,s_*xx,color=PUR,lw=0.9,alpha=0.7)
for s_ in rng.normal(M0,V0**0.5,14): axs[1].plot(xx,s_*xx,color=GREEN,lw=0.9,alpha=0.7)
sdp=np.sqrt(1/A0+xs**2*V0); axs[2].fill_between(xs,M0*xs-2*sdp,M0*xs+2*sdp,color=GL,lw=0); axs[2].plot(xs,M0*xs,color=GREEN,lw=1.8)
for ax in axs: pts(ax,s=26)
axs[0].set_ylabel('$y$'); fig.tight_layout(w_pad=0.5); save(fig,'f1221.png')

# ---- 12.3-1 prior x likelihood = posterior for w
wg=np.linspace(-1.2,2.0,400); nd=lambda w,m,v: np.exp(-(w-m)**2/(2*v))/np.sqrt(2*np.pi*v)
fig,ax=plt.subplots(figsize=(4.3,2.4),dpi=240)
ax.plot(wg,nd(wg,0,1/L0),color=GREY,lw=1.5,ls='--'); ax.plot(wg,nd(wg,float(x@y)/Sxx,1/(A0*Sxx)),color=BLUE,lw=1.5,ls=':'); ax.plot(wg,nd(wg,M0,V0),color=GREEN,lw=2.2)
ax.text(-1.1,0.62,'prior\n(mean 0)',color=GREY,fontsize=9,ha='left',va='center')
ax.text(1.35,1.30,'likelihood\n(peak 0.90)',color=BLUE,fontsize=9,ha='left',va='center')
ax.text(0.72,2.08,'posterior (mean 0.72)',color=GREEN,fontsize=9,ha='center',va='bottom')
ax.text(1.95,0.55,'shrinkage:\n0.90 \u2192 0.72',color=RED,fontsize=9,ha='right',va='center')
ax.set_xlabel('weight  $w$'); ax.set_ylabel('density'); ax.set_ylim(0,2.45); ax.set_xlim(-1.2,2.0); gx(ax); save(fig,'f1231.png')

# ---- 12.4-1 predictive sd decomposition
xs=np.linspace(-9,9,300); fig,ax=plt.subplots(figsize=(4.4,2.4),dpi=240)
ax.axvspan(-3,3,color=LGREY,lw=0); ax.text(0,2.1,'range of\nthe data',ha='center',va='center',fontsize=8.8,color=GREY)
ax.plot(xs,np.sqrt(1/A0+xs**2*V0),color=GREEN,lw=2.2); ax.plot(xs,np.full_like(xs,A0**-0.5),color=GREY,lw=1.4,ls='--'); ax.plot(xs,np.abs(xs)*V0**0.5,color=OR,lw=1.6,ls=':')
ax.text(-8.8,0.88,'noise only:\n$1/\\sqrt{\\alpha}$',color=GREY,fontsize=9,ha='left',va='top'); ax.text(8.8,np.sqrt(1+81*V0)+0.1,'predictive s.d.',color=GREEN,fontsize=9,ha='right',va='bottom')
ax.text(8.8,0.88,'weights\nuncertain',color=OR,fontsize=9,ha='right',va='top')
ax.set_xlim(-9,9); ax.set_ylim(0,2.75); ax.set_xlabel('new setting  $x_*$'); ax.set_ylabel('standard deviation'); gx(ax); save(fig,'f1241.png')

# ---- 12.6-1 flow chart of BayesianRidge.fit
fig,ax=new(3.7,3.9)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=7.7): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
Ar=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.2,7.1,5.6,0.5,'Input:  $\\mathbf{X}$ ($n\\times p$),  $\\mathbf{y}$,  optional sample_weight',fc=LGREY,ec=GREY)
fb(3.2,6.3,5.6,0.55,'Centre $\\mathbf{X}$ and $\\mathbf{y}$ (intercept), apply weights')
fb(3.2,5.4,5.6,0.8,'Start:  $\\alpha = 1/\\mathrm{Var}(\\mathbf{y})$,  $\\lambda = 1$\n(or alpha_init, lambda_init)',fc='#F1EAF7',ec=PUR)
fb(3.2,4.45,5.6,0.55,'One SVD of $\\mathbf{X}$:  eigenvalues $d_j^{2}$ of $\\mathbf{X}^{T}\\mathbf{X}$')
fb(3.2,3.35,5.6,0.8,'Posterior mean $\\mathbf{m}$ (Eq. 12.3-11), then\n$\\gamma$ (12.4-7), new $\\lambda$ and $\\alpha$ (12.4-10)',fc='#E3F4EA',ec=GREEN)
ax.add_patch(Polygon([(3.2,2.6),(5.3,2.0),(3.2,1.4),(1.1,2.0)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.2,2.0,'$\\sum|\\Delta\\mathbf{m}| <$ tol\nor max_iter?',ha='center',va='center',fontsize=7.7,color=NAVY)
fb(3.2,0.75,5.6,0.55,'Final $\\mathbf{m}$ and $\\mathbf{S}$ from the final $\\alpha,\\lambda$; intercept')
fb(3.2,-0.1,5.6,0.55,'Store coef_, intercept_, alpha_, lambda_, sigma_',fc=LGREY,ec=GREY)
Ar((3.2,6.85),(3.2,6.58)); Ar((3.2,6.02),(3.2,5.8)); Ar((3.2,5.0),(3.2,4.73)); Ar((3.2,4.17),(3.2,3.75)); Ar((3.2,2.95),(3.2,2.55))
Ar((3.2,1.45),(3.2,1.03)); ax.text(3.32,1.24,'yes',fontsize=7.7,color=NAVY,va='center'); Ar((3.2,0.47),(3.2,0.18))
ax.plot([5.3,6.4,6.4],[2.0,2.0,3.35],color=NAVY,lw=0.9); Ar((6.4,3.35),(6.0,3.35))
ax.text(5.5,2.1,'no',fontsize=7.7,color=NAVY,va='bottom')
ax.set_xlim(0,6.6); ax.set_ylim(-0.45,7.45); save(fig,'f1261.png')

# ---- diabetes pieces (Chapter 9 split)
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0)
Xc=Xtr-Xtr.mean(0); yc=ytr-ytr.mean(); n,p=Xc.shape; U,sv,Vt=np.linalg.svd(Xc,full_matrices=False); d2=sv**2; Uy=U.T@yc
def post_mean(a,l): return Vt.T@(a*sv*Uy/(l+a*d2))      # posterior mean via SVD
def logev(a,l):
    m=post_mean(a,l); sse=np.sum((yc-Xc@m)**2)
    return 0.5*(p*np.log(l)+n*np.log(a)-a*sse-l*m@m-np.sum(np.log(l+a*d2))-n*np.log(2*np.pi))
def path(a,l,k=40):
    P=[(a,l)]
    for _ in range(k):
        m=post_mean(a,l); g=np.sum(a*d2/(l+a*d2)); sse=np.sum((yc-Xc@m)**2); l=g/(m@m); a=(n-g)/sse; P.append((a,l))
    return np.array(P)
brd=BayesianRidge().fit(Xtr,ytr); a0=1/yc.var(); P=path(a0,1.0,9)
# ---- 12.6-2 evidence surface and the iteration path
lg_l=np.linspace(-8,1,181); lg_a=np.linspace(-5.0,-2.6,181); G=np.array([[logev(10**b_,10**a_) for a_ in lg_l] for b_ in lg_a])
fig,ax=plt.subplots(figsize=(3.5,2.45),dpi=240); mx=G.max()
cs=ax.contourf(lg_l,lg_a,G,levels=mx-np.array([1500,1000,400,200,100,50,20,10,5,2,0.5,-1]),cmap='Purples',alpha=0.85,extend='min'); ax.contour(lg_l,lg_a,G,levels=mx-np.array([1000,400,200,100,50,20,10,5,2,0.5]),colors='white',linewidths=0.5)
ax.plot(np.log10(P[:,1]),np.log10(P[:,0]),'-o',color=NAVY,ms=3.6,lw=1.1,mfc='white',zorder=5)
ax.plot(np.log10(P[0,1]),np.log10(P[0,0]),'s',color=NAVY,ms=6,zorder=6); ax.text(np.log10(P[0,1]),np.log10(P[0,0])+0.1,'start',fontsize=9,color=NAVY,ha='center')
ax.plot(np.log10(brd.lambda_),np.log10(brd.alpha_),'*',color=GREEN,ms=13,mec='white',zorder=7); ax.text(np.log10(brd.lambda_)+0.4,np.log10(brd.alpha_)+0.3,'maximum of\nthe evidence',fontsize=9,color=GREEN,ha='left',va='bottom',weight='bold',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.9))
ax.set_xlabel('weight precision  $\\log_{10}\\lambda$'); ax.set_ylabel('noise precision  $\\log_{10}\\alpha$'); save(fig,'f1262.png')
NUM['evidence']={'start_alpha':float(a0),'path_n':int(len(P)),'alpha_':float(brd.alpha_),'lambda_':float(brd.lambda_),'n_iter':int(brd.n_iter_),'max_logev':float(mx),'logev_start':float(logev(a0,1.0))}

# ---- 12.9-1 ARD sparsity (same data as Listing 12.6)
rng=np.random.default_rng(0); na,pa,hot=100,30,[2,5,11,17]; wt=np.zeros(pa); wt[hot]=[3.0,-2.0,1.5,1.0]
Xa=rng.normal(size=(na+1000,pa)); ya=Xa@wt+rng.normal(0,1.0,na+1000)
ard=ARDRegression().fit(Xa[:na],ya[:na]); ard2=ARDRegression(threshold_lambda=100).fit(Xa[:na],ya[:na]); bra=BayesianRidge().fit(Xa[:na],ya[:na])
fig,axs=plt.subplots(1,2,figsize=(5.4,2.5),dpi=240,gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]; j=np.arange(pa)
ax.bar(j,wt,width=0.55,color=LGREY,ec=GREY,lw=0.6,label='true',zorder=2); ax.plot(j,bra.coef_,'s',color=OR,ms=3.8,mfc='white',mew=1.0,label='BayesianRidge',zorder=4); ax.plot(j,ard2.coef_,'o',color=GREEN,ms=3.8,label='ARDRegression',zorder=5)
ax.axhline(0,color=GREY,lw=0.6); ax.set_xlabel('feature  $j$'); ax.set_ylabel('weight  $w_j$'); ax.legend(fontsize=8.2,loc='upper right',handletextpad=0.3,borderaxespad=0.2); ax.set_title('Weights',fontsize=9.5,color=NAVY); ax.set_xlim(-1,pa); ax.set_ylim(-2.7,3.6)
ax=axs[1]; lam=ard.lambda_; cols=[GREEN if k in hot else '#A9B1BD' for k in j]
ax.bar(j,np.log10(lam)+3,bottom=-3,width=0.7,color=cols); ax.axhline(4,color=PUR,lw=1,ls='--'); ax.axhline(2,color=PUR,lw=1,ls=':')
ax.text(29.5,6.3,'dashed: default threshold $10^4$',fontsize=8.5,color=PUR,va='top',ha='right'); ax.text(29.5,5.55,'dotted: threshold 100',fontsize=8.5,color=PUR,va='top',ha='right')
ax.set_ylim(-1.5,6.4); ax.set_xlim(-1,pa); ax.set_xlabel('feature  $j$'); ax.set_ylabel('$\\log_{10}\\lambda_j$'); ax.set_title('Learned weight precisions',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1291.png')
NUM['ard']={'kept_default':int((ard.lambda_<1e4).sum()),'kept_100':int((ard2.lambda_<100).sum()),'log10_lambda_hot':[round(float(np.log10(v)),2) for v in ard.lambda_[hot]],'min_log10_lambda_rest':round(float(np.log10(np.delete(ard.lambda_,hot).min())),2)}

# ---- 12.10-1 gallery: the weight precision turned
rs=np.random.default_rng(3); xt=np.sort(rs.uniform(0,1,9)); sig=0.2; yt=np.sin(2*np.pi*xt)+rs.normal(0,sig,9)
feat=lambda t: np.vander(2*t-1,8,increasing=True)[:,1:]
xg=np.linspace(0,1,200); Ft=feat(xt); Fg=feat(xg)
fig,axs=plt.subplots(2,3,figsize=(5.4,3.6),dpi=240,sharex=True,sharey=True)
panels=[('fixed',1e-3),('fixed',1e-1),('fixed',1e1),('fixed',1e3),('br',None),('ard',None)]; NUM['gallery']={}
for ax,(kind,lv) in zip(axs.ravel(),panels):
    if kind=='fixed': m=MyBayesRidge(fixed=(1/sig**2,lv)).fit(Ft,yt); title=f'$\\lambda$ = {lv:g}  (fixed)'; tc=PUR; lab=f'fixed_{lv:g}'
    elif kind=='br': m=BayesianRidge().fit(Ft,yt); title=f'evidence: $\\lambda$ = {m.lambda_:.2f}'; tc=GREEN; lab='br'
    else: m=ARDRegression().fit(Ft,yt); title='ARDRegression'; tc=GREEN; lab='ard'
    mu,sd_=m.predict(Fg,return_std=True); ax.fill_between(xg,mu-2*sd_,mu+2*sd_,color=GL,lw=0); ax.plot(xg,np.sin(2*np.pi*xg),color=GREY,lw=1,ls='--'); ax.plot(xg,mu,color=GREEN,lw=1.6); pts(ax,xt,yt,s=16)
    ax.set_title(title,fontsize=8.8,color=tc); gx(ax); ax.set_ylim(-2.3,2.3); ax.set_xlim(0,1)
    NUM['gallery'][lab]={'max_sd':round(float(sd_.max()),3),'min_sd':round(float(sd_.min()),3),'alpha_':round(float(getattr(m,'alpha_',1/sig**2)),2)}
for ax in axs[1]: ax.set_xlabel('$x$')
for ax in axs[:,0]: ax.set_ylabel('$y$')
fig.tight_layout(w_pad=1.1,h_pad=0.6); save(fig,'f12101.png')

# ---- 12.10-2 validation curve: cross-validation against the evidence
ratios=np.logspace(-3,1.6,24); cvm=[]; evm=[]
from scipy.optimize import minimize_scalar
for r_ in ratios:
    sc=cross_val_score(Ridge(alpha=r_),Xtr,ytr,cv=5,scoring='neg_mean_squared_error'); cvm.append(-sc.mean())
    f=lambda la: -logev(10**la,r_*10**la); o=minimize_scalar(f,bounds=(-6,0),method='bounded'); evm.append(-o.fun)
cvm=np.array(cvm); evm=np.array(evm); rb=brd.lambda_/brd.alpha_; rcv=ratios[np.argmin(cvm)]; rev=ratios[np.argmax(evm)]
fig,axs=plt.subplots(1,2,figsize=(4.7,2.3),dpi=240)
ax=axs[0]; ax.semilogx(ratios,cvm,'-o',color=GREY,ms=3,lw=1.4); ax.axvline(rb,color=GREEN,lw=1.4); ax.text(rb*1.25,cvm.max()*0.985,'evidence\nchoice',color=GREEN,fontsize=8.8,va='top'); ax.set_xlabel('ridge strength  $\\lambda/\\alpha$'); ax.set_ylabel('5-fold CV error (MSE)'); ax.set_title('Held-out error',fontsize=9.5,color=NAVY); gx(ax)
ax=axs[1]; ax.semilogx(ratios,evm,'-o',color=PUR,ms=3,lw=1.4); ax.axvline(rb,color=GREEN,lw=1.4); ax.set_xlabel('ridge strength  $\\lambda/\\alpha$'); ax.set_ylabel('log evidence'); ax.set_title('Evidence (best $\\alpha$ for each)',fontsize=9.5,color=NAVY); gx(ax)
fig.tight_layout(w_pad=1.0); save(fig,'f12102.png')
NUM['valid']={'ratio_evidence_BR':float(rb),'ratio_best_cv':float(rcv),'ratio_best_profile_evidence':float(rev),'cv_mse_at_best':float(cvm.min()),'cv_mse_at_ratio_BR':float(cross_val_score(Ridge(alpha=rb),Xtr,ytr,cv=5,scoring='neg_mean_squared_error').mean()*-1)}

# ---- 12.11-1 case study: intervals and weights
ols=LinearRegression().fit(Xtr,ytr); ardd=ARDRegression().fit(Xtr,ytr)
mean,sdp=brd.predict(Xte,return_std=True); z=np.abs(yte-mean)/sdp
o=np.argsort(mean)[::3]; k=np.arange(len(o))     # every third test patient, sorted by prediction
fig,axs=plt.subplots(1,2,figsize=(5.5,2.9),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; out=z[o]>=2
ax.fill_between(k,mean[o]-2*sdp[o],mean[o]+2*sdp[o],color=GL,lw=0); ax.plot(k,mean[o],color=GREEN,lw=1.4)
ax.scatter(k[~out],yte[o][~out],s=12,color=BLUE,lw=0,zorder=4); ax.scatter(k[out],yte[o][out],s=22,color=RED,marker='x',lw=1.2,zorder=5)
ax.set_xlabel('test patients, sorted by prediction'); ax.set_ylabel('progression'); ax.set_title('Predicted mean $\\pm$ 2 s.d.',fontsize=9.5,color=NAVY); gx(ax); ax.set_xlim(0,len(o)-1)
ax=axs[1]; names=np.array(d.feature_names); order=np.argsort(ols.coef_); yy=np.arange(10)
ax.hlines(yy,np.minimum(ols.coef_[order],brd.coef_[order]),np.maximum(ols.coef_[order],brd.coef_[order]),color=LGREY,lw=2.2,zorder=1)
ax.plot(ols.coef_[order],yy,'s',color=GREY,ms=4.2,mfc='white',label='least squares',zorder=3); ax.plot(brd.coef_[order],yy,'o',color=GREEN,ms=4.2,label='BayesianRidge',zorder=4); ax.plot(ardd.coef_[order],yy,'^',color=OR,ms=4.2,label='ARDRegression',zorder=5)
ax.set_yticks(yy); ax.set_yticklabels(names[order]); ax.axvline(0,color=GREY,lw=0.6); ax.legend(fontsize=8.2,loc='lower right',handletextpad=0.2,borderaxespad=0.1); ax.set_xlabel('weight  $w_j$'); ax.set_title('Fitted weights',fontsize=9.5,color=NAVY); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); save(fig,'f12111.png')
NUM['case']={'cover1':float(np.mean(z<1)),'cover2':float(np.mean(z<2)),'n_test':int(len(z)),'sd_min':float(sdp.min()),'sd_max':float(sdp.max()),'noise_sd':float(brd.alpha_**-0.5),'n_outside_shown':int(out.sum()),'n_shown':int(len(o)),
  'param_share_of_var_max':float(np.max(1-(1/brd.alpha_)/(sdp**2))),'param_share_of_var_mean':float(np.mean(1-(1/brd.alpha_)/(sdp**2))),'rmse_pred_mean_sd':float(sdp.mean())}
NUM['coefs']={'features':list(d.feature_names),'ols':[round(float(v),1) for v in ols.coef_],'br':[round(float(v),1) for v in brd.coef_],'ard':[round(float(v),1) for v in ardd.coef_]}
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

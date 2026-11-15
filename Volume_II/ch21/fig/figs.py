# All figures of Chapter 21. Run: python3 fig/figs.py  (PNGs into ch21/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.datasets import load_diabetes, make_friedman1
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grd(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx,yy,s=24,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.7,zorder=5,**k)
# ---- shared data: a noisy sine, 28 points
rng=np.random.default_rng(4); xs=np.sort(rng.uniform(0,6,28)); ys=np.sin(xs)+0.25*rng.normal(size=28); grid=np.linspace(0,6,300)
def krr_fit(x,y,g,a,kernel='rbf'):
    m=KernelRidge(alpha=a,kernel=kernel,gamma=g).fit(x[:,None],y-y.mean()); return lambda t: m.predict(t[:,None])+y.mean()
def df_of(x,g,a):
    ev=np.linalg.eigvalsh(rbf_kernel(x[:,None],gamma=g)); return float((ev/(ev+a)).sum())
# toy
Xt=np.array([-1.,0,1]); yt=np.array([1.,0,1]); G0,A0=0.5,0.1
Kt=rbf_kernel(Xt[:,None],gamma=G0); at=np.linalg.solve(Kt+A0*np.eye(3),yt); NUM['toy_a']=[round(float(v),4) for v in at]

# ---- 21.0-1 opener: ridge on x versus kernel ridge
fig,axs=plt.subplots(1,2,figsize=(5.6,2.1),dpi=240,sharey=True)
rd=Ridge(alpha=1.0).fit(xs[:,None],ys); f=krr_fit(xs,ys,0.5,0.1)
NUM['open']={'ridge_r2':round(float(rd.score(xs[:,None],ys)),3),'krr_r2':round(float(1-np.sum((ys-f(xs))**2)/np.sum((ys-ys.mean())**2)),3),
  'ridge_rmse_truth':round(float(np.sqrt(np.mean((rd.predict(grid[:,None])-np.sin(grid))**2))),3),'krr_rmse_truth':round(float(np.sqrt(np.mean((f(grid)-np.sin(grid))**2))),3)}
for ax,t,pred,col in zip(axs,['Ridge on the raw feature','Kernel ridge, RBF kernel'],[rd.predict(grid[:,None]),f(grid)],[GREEN,GREEN]):
    ax.plot(grid,np.sin(grid),color=GREY,lw=1,ls=':',zorder=2); ax.plot(grid,pred,color=col,lw=2.1,zorder=3); pts(ax,xs,ys,s=26); grd(ax)
    ax.set_title(t,fontsize=9.5,color=NAVY); ax.set_xlabel('feature  $x$')
axs[0].set_ylabel('target  $y$'); axs[0].text(0.15,-1.55,'dotted: the curve behind the noise',fontsize=8.5,color=GREY); axs[0].set_ylim(-1.8,1.6)
fig.tight_layout(w_pad=0.8); sv(fig,'f2100.png',460)

# ---- 21.1-1 the toy data with the best line
fig,ax=plt.subplots(figsize=(3.7,2.3),dpi=240)
xx=np.linspace(-1.8,1.8,50); ax.axhline(2/3,color=GREEN,lw=2,zorder=3); pts(ax,Xt,yt,s=46)
for a,b in zip(Xt,yt): ax.text(a+0.08,b+(0.1 if b else 0.1),f'({a:.0f}, {b:.0f})'.replace('-','−'),fontsize=8.5,color=GREY)
ax.axvline(0.5,color=PUR,lw=1,ls='--'); ax.text(0.56,0.05,'$x$ = 0.5:\nwhat is $y$?',color=PUR,fontsize=9)
ax.text(-1.75,0.55,'best straight line: flat',color=GREEN,fontsize=8.5,weight='bold'); ax.set_xlim(-1.8,1.8); ax.set_ylim(-0.2,1.35); grd(ax)
ax.set_xlabel('offset $x$'); ax.set_ylabel('response $y$'); sv(fig,'f2111.png',295)

# ---- 21.2-1 a new column straightens a curve
rng2=np.random.default_rng(7); xp=np.linspace(-2,2,12); yp=xp**2+rng2.normal(0,0.25,12)
fig,axs=plt.subplots(1,3,figsize=(5.6,2.0),dpi=240)
ax=axs[0]; ln=LinearRegression().fit(xp[:,None],yp); pts(ax,xp,yp,s=18); ax.plot([-2.3,2.3],ln.predict([[-2.3],[2.3]]),color=GREEN,lw=1.8); grd(ax)
ax.set_title('Against $x$:\nno line fits',fontsize=9,color=NAVY); ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax=axs[1]; l2=LinearRegression().fit((xp**2)[:,None],yp); pts(ax,xp**2,yp,s=18); ax.plot([0,4.3],l2.predict([[0],[4.3]]),color=GREEN,lw=1.8); grd(ax)
ax.set_title('Against the new\ncolumn $x^2$: a line',fontsize=9,color=NAVY); ax.set_xlabel('$x^2$')
ax=axs[2]; tt=np.linspace(-2.3,2.3,100); pts(ax,xp,yp,s=18); ax.plot(tt,l2.predict((tt**2)[:,None]),color=GREEN,lw=1.8); grd(ax)
ax.set_title('Back against $x$:\na curve',fontsize=9,color=NAVY); ax.set_xlabel('$x$')
for ax in axs: ax.set_ylim(-0.7,4.9)
axs[1].set_yticklabels([]); axs[2].set_yticklabels([]); fig.tight_layout(w_pad=0.5); sv(fig,'f2121.png',460)
NUM['f212']={'r2_x':round(float(ln.score(xp[:,None],yp)),3),'r2_x2':round(float(l2.score((xp**2)[:,None],yp)),3)}

# ---- 21.3-1 kernel profiles and Gram matrices
fig=plt.figure(figsize=(5.6,1.95),dpi=240); gs=fig.add_gridspec(1,4,width_ratios=[1.55,1,1,1],wspace=0.32)
ax=fig.add_subplot(gs[0]); d=np.linspace(-3,3,300)
for g,c,ls in [(0.2,PUR,'-'),(1,GREEN,'-'),(5,OR,'-')]: ax.plot(d,np.exp(-g*d**2),color=c,lw=1.6,ls=ls,label=f'$\\gamma$ = {g:g}')
ax.plot(d,np.exp(-np.abs(d)),color=GREY,lw=1.3,ls='--',label='Laplacian'); ax.legend(loc='upper center',ncol=2,fontsize=7.8,handlelength=1.3,columnspacing=0.8,borderaxespad=0.1,title='RBF, and Laplacian ($\\gamma$ = 1)',title_fontsize=7.8); grd(ax)
ax.set_xlabel('$x - z$'); ax.set_ylabel('$k(x,z)$'); ax.set_ylim(0,1.62); ax.set_yticks([0,0.5,1]); ax.set_xlim(-3,3)
xg=np.sort(np.random.default_rng(1).uniform(0,6,12))
NUM['gram']={}
for j,g in enumerate([0.02,0.5,20]):
    ax=fig.add_subplot(gs[1+j]); Kg=rbf_kernel(xg[:,None],gamma=g); ax.imshow(Kg,cmap='Blues',vmin=0,vmax=1); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True)
    ax.set_title(f'$\\gamma$ = {g:g}',fontsize=9,color=NAVY); NUM['gram'][str(g)]={'min':round(float(Kg.min()),4),'offdiag_mean':round(float((Kg.sum()-12)/132),4),'min_eig':float(np.linalg.eigvalsh(Kg).min())}
sv(fig,'f2131.png',450)

# ---- 21.4-1 the model is a sum of bumps (toy)
fig,ax=plt.subplots(figsize=(4.4,2.6),dpi=240); tt=np.linspace(-2.6,2.6,400)
cols=[BLUE,PUR,BLUE]; tot=np.zeros_like(tt)
for xi,ai,c in zip(Xt,at,[BLUE,OR,BLUE]):
    b=ai*np.exp(-G0*(tt-xi)**2); tot+=b; ax.plot(tt,b,color=c,lw=1.2,ls='-' if ai>0 else '--',zorder=2)
ax.plot(tt,tot,color=GREEN,lw=2.3,zorder=4); pts(ax,Xt,yt,s=44); ax.axhline(0,color=GREY,lw=0.6)
ax.text(-2.55,1.95,'$a_1 k(x_1, x)$',color=BLUE,fontsize=8.8); ax.text(1.0,1.95,'$a_3 k(x_3, x)$',color=BLUE,fontsize=8.8); ax.text(-2.55,-1.3,'$a_2 k(x_2, x)$:\na negative bump',color=OR,fontsize=8.8,va='top')
ax.text(1.95,0.3,'$f(x)$',color=GREEN,fontsize=10,weight='bold',ha='left')
ax.set_xlim(-2.6,2.6); ax.set_ylim(-2.25,2.15); grd(ax); ax.set_xlabel('$x$'); ax.set_ylabel('bumps and their sum')
NUM['toy_f']={'f_at_data':[round(float(v),4) for v in rbf_kernel(Xt[:,None],gamma=G0)@at]}
sv(fig,'f2141.png',332)

# ---- 21.6-1 flow chart of fit and predict
fig,ax=new(4.8,3.75)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
cx,rx=1.95,5.75
fb(cx,5.0,3.5,0.55,'fit($\\mathbf{X}$, $\\mathbf{y}$):  validate the inputs',fc=LGREY,ec=GREY)
fb(cx,3.9,3.5,1.0,'Gram matrix  $\\mathbf{K}=k(\\mathbf{X},\\mathbf{X})$\n($n^2p$ operations; for "precomputed"\n$\\mathbf{X}$ is already $\\mathbf{K}$)',fc='#E3F4EA',ec=GREEN)
fb(cx,2.85,3.5,0.55,'add $\\alpha$ to the diagonal:  $\\mathbf{K}+\\alpha\\mathbf{I}$')
fb(cx,1.75,3.5,1.0,'solve  $(\\mathbf{K}+\\alpha\\mathbf{I})\\,\\mathbf{a}=\\mathbf{y}$  (Eq. 21.4-3)\nCholesky, $n^3/3$ operations;\nleast squares if it fails')
fb(cx,0.7,3.5,0.55,'store  dual_coef_ = $\\mathbf{a}$,  X_fit_ = $\\mathbf{X}$',fc=LGREY,ec=GREY)
fb(rx,5.0,3.0,0.55,'predict($\\mathbf{X}_{\\mathrm{new}}$)',fc=LGREY,ec=GREY)
fb(rx,3.9,3.0,1.0,'kernel values\n$\\mathbf{K}(\\mathbf{X}_{\\mathrm{new}},\\mathbf{X}_{\\mathrm{fit}})$\n($np$ operations per sample)',fc='#E3F4EA',ec=GREEN)
fb(rx,2.5,3.0,0.95,'$f(\\mathbf{x})=\\sum_i a_i\\,k(\\mathbf{x}_i,\\mathbf{x})$\n= $\\mathbf{K}(\\mathbf{X}_{\\mathrm{new}},\\mathbf{X})\\,\\mathbf{a}$',fc='#E3F4EA',ec=GREEN)
A((cx,4.72),(cx,4.4)); A((cx,3.4),(cx,3.125)); A((cx,2.575),(cx,2.25)); A((cx,1.25),(cx,0.975))
A((rx,4.72),(rx,4.4)); A((rx,3.4),(rx,2.975))
ax.add_patch(FancyArrowPatch((cx+1.75,0.7),(rx-1.5,3.7),arrowstyle='-|>',mutation_scale=8,color=PUR,lw=0.9,ls=(0,(3,2)),connectionstyle='arc3,rad=0.28'))
ax.text(5.75,1.1,'the stored $\\mathbf{a}$ and $\\mathbf{X}$\nserve every later call',ha='center',va='center',fontsize=8.3,color=PUR,style='italic')
ax.set_xlim(-0.05,7.45); ax.set_ylim(0.3,5.4); sv(fig,'f2161.png',450)

# ---- 21.9-1 galleries over gamma and alpha
fig,axs=plt.subplots(2,4,figsize=(5.3,3.1),dpi=240,sharex=True,sharey=True); NUM['gal']={'gamma':[],'alpha':[]}
for row,(vals,lab,key) in enumerate([([0.02,0.5,5,100],'$\\gamma$',0),([1e-5,0.1,10,1000],'$\\alpha$',1)]):
    for ax,v in zip(axs[row],vals):
        g,a=(v,0.1) if key==0 else (0.5,v); f=krr_fit(xs,ys,g,a); p=f(grid); rm=float(np.sqrt(np.mean((p-np.sin(grid))**2))); dfv=df_of(xs,g,a)
        NUM['gal']['gamma' if key==0 else 'alpha'].append({'value':v,'rmse_truth':round(rm,3),'df':round(dfv,2),'max_abs':round(float(np.abs(p).max()),2)})
        ax.plot(grid,np.sin(grid),color=GREY,lw=0.9,ls=':'); ax.plot(grid,np.clip(p,-3,3),color=GREEN,lw=1.7,zorder=3); pts(ax,xs,ys,s=10); grd(ax)
        e=int(np.floor(np.log10(v))); ttl=f'{lab} = {v:g}' if (v>=1e-3 and v<1e4) else f'{lab} = $10^{{{e}}}$'
        ax.set_title(ttl,fontsize=9,color=PUR); ax.text(0.04,0.04,f'df = {dfv:.1f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,bbox=dict(fc='white',ec='none',pad=0.8,alpha=0.85))
        ax.set_ylim(-1.9,1.9); ax.set_xticks([0,3,6]); ax.tick_params(labelsize=8.5)
for ax in axs[:,0]: ax.set_ylabel('$y$',fontsize=9)
for ax in axs[1]: ax.set_xlabel('$x$',fontsize=9)
fig.tight_layout(h_pad=0.6,w_pad=0.3); sv(fig,'f2191.png',442)

# ---- 21.9-2 validation heat map and curve
als=10.0**np.arange(-6,2.01,0.5); gms=10.0**np.arange(-2,2.01,0.5)
def cv_rmse(g,a,k=5):
    err=[];tr=[]
    for itr,ite in KFold(k,shuffle=True,random_state=0).split(xs):
        f=krr_fit(xs[itr],ys[itr],g,a); err.append(np.mean((f(xs[ite])-ys[ite])**2)); tr.append(np.mean((f(xs[itr])-ys[itr])**2))
    return float(np.sqrt(np.mean(err))),float(np.sqrt(np.mean(tr)))
CV=np.array([[cv_rmse(g,a)[0] for g in gms] for a in als]); ia,ig=np.unravel_index(CV.argmin(),CV.shape)
NUM['cv']={'best_alpha':float(als[ia]),'best_gamma':float(gms[ig]),'best_rmse':round(float(CV.min()),4),'noise_sd':0.25,'default_gamma_note':'1/n_features=1'}
fig,axs=plt.subplots(1,2,figsize=(5.4,2.5),dpi=240,gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]; im=ax.pcolormesh(np.log10(gms),np.log10(als),np.log10(np.minimum(CV,2.0)),cmap='Blues_r',shading='nearest',vmin=np.log10(0.25),vmax=np.log10(1.0))
ax.plot(np.log10(gms[ig]),np.log10(als[ia]),'*',color=OR,ms=11,mec='white',mew=0.7); ax.set_xlabel('$\\log_{10}\\gamma$'); ax.set_ylabel('$\\log_{10}\\alpha$'); ax.set_title('Cross-validated RMSE',fontsize=9,color=NAVY)
cb=fig.colorbar(im,ax=ax,pad=0.03,fraction=0.05); cb.set_ticks(np.log10([0.25,0.5,1.0])); cb.set_ticklabels(['0.25','0.5','≥ 1']); cb.ax.tick_params(labelsize=8.5)
ax=axs[1]; a0=float(als[ia]); r=np.array([cv_rmse(g,a0) for g in gms])
ax.plot(np.log10(gms),r[:,1],color=BLUE,lw=1.6,marker='o',ms=3,label='training folds'); ax.plot(np.log10(gms),r[:,0],color=GREY,lw=1.6,marker='s',ms=3,label='validation folds')
ax.axvline(np.log10(gms[ig]),color=PUR,lw=1,ls='--'); ax.axhline(0.25,color=RED,lw=0.8,ls=':'); ax.text(1.95,0.265,'noise level 0.25',color=RED,fontsize=8.5,ha='right',va='bottom')
ax.set_ylim(0,0.95); ax.set_xlabel('$\\log_{10}\\gamma$'); ax.set_ylabel('RMSE'); grd(ax); ax.legend(loc='upper right',fontsize=8.5,handlelength=1.5,title=f'$\\alpha$ = {a0:.2g}',title_fontsize=8.5); ax.set_title('Validation curve in $\\gamma$',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=0.8); sv(fig,'f2192.png',448)
NUM['cv']['curve']={'gamma':[float(v) for v in gms],'train':[round(float(v),3) for v in r[:,1]],'valid':[round(float(v),3) for v in r[:,0]]}

# ---- 21.11 case studies (same grids as code/case_diabetes.py and code/case_more.py)
d=load_diabetes(); X,y=d.data,d.target; Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=0)
def model(kernel,scale=True):
    pipe=make_pipeline(StandardScaler() if scale else 'passthrough',KernelRidge(kernel=kernel)); return TransformedTargetRegressor(pipe,transformer=StandardScaler())
p='regressor__kernelridge__'; als2=np.logspace(-2,2,9); gms2=np.logspace(-3.5,-0.5,7)
grids={'linear':{p+'alpha':np.logspace(-1,4,11)},'poly':{p+'alpha':als2,p+'degree':[2,3],p+'gamma':[0.01,0.03,0.1]},'rbf':{p+'alpha':als2,p+'gamma':gms2},'laplacian':{p+'alpha':als2,p+'gamma':gms2}}
R2={'OLS':LinearRegression().fit(Xtr,ytr).score(Xte,yte),'ridge':RidgeCV(alphas=np.logspace(-3,1,41)).fit(Xtr,ytr).score(Xte,yte)}
for k,gr in grids.items(): R2['KRR '+k]=GridSearchCV(model(k),gr,cv=5,scoring='r2').fit(Xtr,ytr).score(Xte,yte)
NUM['diab']={k:round(float(v),3) for k,v in R2.items()}
Xf,yf=make_friedman1(1300,noise=1.0,random_state=0); Xa,Xb,ya,yb=Xf[:300],Xf[300:],yf[:300],yf[300:]
rg=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Xa,ya); gs=GridSearchCV(model('rbf'),{p+'alpha':np.logspace(-4,0,9),p+'gamma':np.logspace(-3,0,7)},cv=5).fit(Xa,ya)
NUM['fried']={'ridge':round(float(rg.score(Xb,yb)),3),'rbf':round(float(gs.score(Xb,yb)),3)}
fig,axs=plt.subplots(1,2,figsize=(5.4,2.7),dpi=240,gridspec_kw={'width_ratios':[1,1.05]})
ax=axs[0]; names=list(R2); vals=[R2[k] for k in names]; cols=[GREY,GREY,GREEN,GREEN,GREEN,GREEN]
ax.barh(range(len(names))[::-1],vals,color=cols,height=0.62)
for yy,v in zip(range(len(names))[::-1],vals): ax.text(v+0.004,yy,f'{v:.3f}',va='center',fontsize=8.5)
ax.set_yticks(range(len(names))[::-1]); ax.set_yticklabels(names,fontsize=8.5); ax.set_xlim(0.28,0.42); ax.set_xlabel('test $R^2$'); ax.set_title('Diabetes: all about equal',fontsize=9,color=NAVY); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax=axs[1]; ax.plot([-4,25],[-4,25],color=GREY,lw=0.9,ls=':'); ax.scatter(yb,rg.predict(Xb),s=7,color=OR,marker='s',alpha=0.55,lw=0,label=f'ridge  ($R^2$ = {rg.score(Xb,yb):.3f})'); ax.scatter(yb,gs.predict(Xb),s=7,color=GREEN,alpha=0.6,lw=0,label=f'RBF kernel  ($R^2$ = {gs.score(Xb,yb):.3f})')
ax.set_xlim(-3,25); ax.set_ylim(-3,25); ax.set_aspect('equal'); ax.set_xlabel('measured'); ax.set_ylabel('predicted'); ax.set_title('Friedman #1: kernels help',fontsize=9,color=NAVY); grd(ax)
ax.legend(loc='lower right',fontsize=8.5,handletextpad=0.2,markerscale=1.8,borderaxespad=0.1)
fig.tight_layout(w_pad=0.8); sv(fig,'f21111.png',436)

with open('verify/fig_numbers.json','w') as _f: json.dump({'num':NUM,'fonts':CHK},_f,indent=1)
print(json.dumps(NUM)[:3000]); print({k:v['min_font_printed'] for k,v in CHK.items()})

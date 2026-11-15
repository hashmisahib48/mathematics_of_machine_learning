# All figures of Chapter 11. Run: python3 fig/figs.py  (PNGs go into ch11/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error')
from figstyle import *
from matplotlib.patches import Ellipse
from sklearn.linear_model import (Lasso, ElasticNet, LinearRegression, Ridge, lasso_path, enet_path, lars_path, LassoCV,
                                  ElasticNetCV, OrthogonalMatchingPursuit, MultiTaskLasso, LassoLarsIC)
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
NUM={}
if os.path.exists('verify/fig_numbers.json'):
    with open('verify/fig_numbers.json') as fh: NUM=json.load(fh)
ONLY=set(sys.argv[1:])
def want(k): return (not ONLY) or (k in ONLY)
# ---- the toy of the chapter: delivery data with traffic lights
dist=np.array([1.,2,3,4,5]); lights=np.array([2.,1,4,3,5]); t=np.array([3.,4,6,7,10])
Xt=np.c_[dist,lights]; Xtc=Xt-Xt.mean(0); ytc=t-t.mean(); n5=5
G=Xtc.T@Xtc/n5; cvec=Xtc.T@ytc/n5; what=np.linalg.solve(G,cvec)
def grid(ax):
    ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ---- 11.0-1 opener: ellipses touching a diamond and a circle
if want('f1100'):
    fig,axs=plt.subplots(1,2,figsize=(5.0,2.55),dpi=240,sharey=True)
    th=np.linspace(0,2*np.pi,400)
    ev,U=np.linalg.eigh(G)           # ellipse (w-what)'G(w-what)/2 = L  -> semi axes sqrt(2L/ev)
    # ridge solution on circle radius 1
    from scipy.optimize import brentq
    def wr(lam): return np.linalg.solve(G+lam*np.eye(2),G@what)
    lam=brentq(lambda l:np.linalg.norm(wr(l))-1,1e-6,50); wring=wr(lam)
    NUM['opener']={'ridge_point':[round(float(v),3) for v in wring],'what':[round(float(v),4) for v in what]}
    for ax,kind in zip(axs,['l1','l2']):
        sol=np.array([1.0,0.0]) if kind=='l1' else wring
        Ls=0.5*(sol-what)@G@(sol-what)
        for f in [0.25,0.6,1.0,1.7]:
            L=Ls*f; a=np.sqrt(2*L/ev); pts_=(U@(np.vstack([a[0]*np.cos(th),a[1]*np.sin(th)])))+what[:,None]
            ax.plot(pts_[0],pts_[1],color=RED,lw=1.3 if f==1.0 else 0.8,alpha=1 if f==1.0 else 0.65)
        if kind=='l1':
            ax.fill([1,0,-1,0],[0,1,0,-1],color='#F1EAF7',zorder=0); ax.plot([1,0,-1,0,1],[0,1,0,-1,0],color=PUR,lw=1.6)
            ax.set_title('L1 penalty: diamond',fontsize=10,color=NAVY)
        else:
            ax.fill(np.cos(th),np.sin(th),color='#F1EAF7',zorder=0); ax.plot(np.cos(th),np.sin(th),color=PUR,lw=1.6)
            ax.set_title('L2 penalty: circle',fontsize=10,color=NAVY)
        ax.plot(*what,'*',color=NAVY,ms=10,zorder=6); ax.plot(*sol,'o',color=GREEN,ms=7,mec='white',mew=0.8,zorder=7)
        ax.axhline(0,color=GREY,lw=0.5,zorder=0); ax.axvline(0,color=GREY,lw=0.5,zorder=0)
        ax.set_aspect('equal'); ax.set_xlim(-1.3,2.7); ax.set_ylim(-1.55,1.75); ax.set_xlabel('$w_1$'); 
    axs[0].set_ylabel('$w_2$')
    bb=dict(fc='white',ec='none',pad=1.2,alpha=0.9)
    for ax in axs: ax.annotate('least squares',xy=what,xytext=(0.95,1.4),fontsize=9,color=NAVY,bbox=bb,arrowprops=dict(arrowstyle='-',color=NAVY,lw=0.6))
    axs[0].annotate('$w_2 = 0$ exactly',xy=(1,0),xytext=(0.7,-1.35),fontsize=9,color=GREEN,bbox=bb,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6))
    axs[1].annotate('both weights\nnon-zero',xy=wring,xytext=(1.0,-1.35),fontsize=9,color=GREEN,bbox=bb,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6))
    fig.tight_layout(w_pad=0.5); save(fig,'f1100.png')

# ---- 11.1-1 the toy data: two panels
if want('f1111'):
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240,sharey=True)
    for ax,xx,lab in zip(axs,[dist,lights],['distance $x_1$ (km)','traffic lights $x_2$']):
        ax.scatter(xx,t,s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5); grid(ax); ax.set_xlabel(lab)
        ax.set_xlim(0,6.4); ax.set_ylim(0,12.5)
    axs[0].set_ylabel('time $y$ (min)')
    r12=float(np.corrcoef(dist,lights)[0,1]); NUM['toy_corr']=round(r12,3)
    axs[1].text(0.2,11.3,f'correlation of $x_1$, $x_2$ = {r12:.2f}',fontsize=8.5,color=PUR)
    fig.tight_layout(w_pad=0.6); save(fig,'f1111.png')

# ---- 11.2-1 sparse recovery: true, OLS, ridge, lasso
if want('f1121'):
    rng=np.random.default_rng(4); n_,p_=50,10
    Xs=rng.normal(size=(n_,p_)); wtrue=np.zeros(p_); wtrue[[0,2,6]]=[3.0,-2.0,1.5]; ys=Xs@wtrue+rng.normal(0,2.0,n_)
    ols_=LinearRegression().fit(Xs,ys).coef_; rid_=Ridge(alpha=20.0).fit(Xs,ys).coef_; las_=Lasso(alpha=0.35).fit(Xs,ys).coef_
    NUM['sparse']={'ols':[round(float(v),2) for v in ols_],'ridge':[round(float(v),2) for v in rid_],'lasso':[round(float(v),2) for v in las_],
                   'n_zero_lasso':int((las_==0).sum()),'n_zero_ridge':int((rid_==0).sum()),'n_zero_ols':int((ols_==0).sum())}
    fig,ax=plt.subplots(figsize=(4.9,2.3),dpi=240)
    xs=np.arange(1,p_+1)
    ax.axhline(0,color=GREY,lw=0.7)
    for j in xs: ax.axvline(j,color=GRID,lw=0.4,zorder=0)
    ax.scatter(xs-0.24,ols_,marker='o',s=26,color=BLUE,label='least squares',zorder=4)
    ax.scatter(xs-0.08,rid_,marker='s',s=24,color=OR,label='ridge',zorder=4)
    ax.scatter(xs+0.08,las_,marker='^',s=30,color=GREEN,label='lasso',zorder=5)
    ax.scatter(xs+0.24,wtrue,marker='_',s=60,color=NAVY,lw=1.6,label='true weight',zorder=4)
    ax.set_xticks(xs); ax.set_xlabel('feature $j$'); ax.set_ylabel('weight $w_j$'); ax.set_ylim(-3.3,4.6)
    ax.legend(ncol=4,fontsize=8.5,loc='upper right',handletextpad=0.2,columnspacing=0.8,borderpad=0.2)
    fig.tight_layout(); save(fig,'f1121.png')

# ---- 11.3-1 soft threshold
if want('f1131'):
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.4),dpi=240)
    ax=axs[0]; w_=np.linspace(-2.2,3.6,400); al=1.0
    for z,c_,lab in [(0.6,OR,'$c = 0.6$'),(3.0,BLUE,'$c = 3$')]:
        J=0.5*w_**2-z*w_+al*np.abs(w_); ax.plot(w_,J,color=c_,lw=1.6,label=lab)
        wm=np.sign(z)*max(abs(z)-al,0); ax.plot(wm,0.5*wm**2-z*wm+al*abs(wm),'o',color=GREEN,ms=6,mec='white',mew=0.8,zorder=6)
    ax.set_xlabel('$w$'); ax.set_ylabel('$J(w)$ + constant'); ax.set_ylim(-3.2,4.5); grid(ax); ax.legend(fontsize=8.5,loc='upper right',borderpad=0.2)
    ax.set_title('$v=1$, $\\alpha=1$',fontsize=9,color=NAVY)
    ax=axs[1]; z=np.linspace(-4,4,401); al=1.0
    soft=np.sign(z)*np.maximum(np.abs(z)-al,0)
    ax.plot(z,z,color=BLUE,lw=1.0,ls=':',label='least squares')
    ax.plot(z,np.where(np.abs(z)>al,z,0),color=GREY,lw=1.2,ls='--',label='hard threshold')
    ax.plot(z,z/(1+al),color=OR,lw=1.4,ls='-.',label='ridge, $c/(1+\\alpha)$')
    ax.plot(z,soft,color=GREEN,lw=2.0,label='lasso, $S(c,\\alpha)$')
    ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
    ax.annotate('flat piece:\nexact zero',xy=(0.2,0),xytext=(0.6,-2.6),fontsize=8.5,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6))
    ax.set_xlabel('$c$ (least-squares weight)'); ax.set_ylabel('fitted weight $w$'); grid(ax); ax.legend(fontsize=8.2,loc='upper left',borderpad=0.2,labelspacing=0.25)
    ax.set_title('three rules, $v = 1$',fontsize=9,color=NAVY); ax.set_xlim(-4,4); ax.set_ylim(-4,4)
    fig.tight_layout(w_pad=0.8); save(fig,'f1131.png')

# ---- 11.4-1 elastic-net unit balls
if want('f1141'):
    fig,axs=plt.subplots(1,4,figsize=(4.9,1.55),dpi=240,sharey=True)
    th=np.linspace(0,2*np.pi,721)
    for ax,rho in zip(axs,[1.0,0.7,0.3,0.0]):
        # penalty rho*|w|_1 + (1-rho)/2*|w|_2^2 = const through the points (+-1,0): value rho+(1-rho)/2
        Lc=rho+(1-rho)/2; xs_=[];ys_=[]
        for a in th:
            d=np.array([np.cos(a),np.sin(a)])
            # solve rho*r*|d|_1 + (1-rho)/2*r^2 = Lc for r>0
            A=(1-rho)/2; Bq=rho*np.abs(d).sum()
            r_=(Lc/Bq) if A==0 else (-Bq+np.sqrt(Bq*Bq+4*A*Lc))/(2*A)
            xs_.append(r_*d[0]); ys_.append(r_*d[1])
        ax.fill(xs_,ys_,color='#F1EAF7'); ax.plot(xs_,ys_,color=PUR,lw=1.5); ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
        ax.set_aspect('equal'); ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.4,1.4); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
        ax.set_title(f'$\\rho$ = {rho:g}',fontsize=9.5,color=NAVY); ax.set_xlabel('$w_1$',fontsize=9)
    axs[0].set_ylabel('$w_2$',fontsize=9); axs[0].text(0,-1.3,'lasso',ha='center',fontsize=8.5,color=GREEN); axs[3].text(0,-1.3,'ridge',ha='center',fontsize=8.5,color=OR)
    fig.tight_layout(w_pad=0.3); save(fig,'f1141.png')

# ---- 11.5-1 path of the toy
if want('f1151'):
    amax=float(np.abs(cvec).max()); als=np.linspace(amax,0,400)
    def wpath(a,rho=1.0):
        # exact lasso path via sklearn on a fine grid
        pass
    al_,co_,_=lasso_path(Xtc,ytc,alphas=np.linspace(3.4,0.0,171)[:-1].tolist()+[1e-4]) if False else (None,None,None)
    grid_a=np.concatenate([np.linspace(3.5,0.02,300)])
    a_,c_,_=lasso_path(Xtc,ytc,alphas=grid_a,tol=1e-10)
    fig,ax=plt.subplots(figsize=(4.4,2.4),dpi=240)
    ax.plot(a_,c_[0],color=BLUE,lw=1.8,label='$w_1$ (distance)'); ax.plot(a_,c_[1],color=OR,lw=1.8,ls='--',label='$w_2$ (lights)')
    ax.axvline(3.4,color=GRID,lw=0.8); ax.axvline(1.4,color=GRID,lw=0.8)
    ax.text(3.3,0.1,'$\\alpha_{\\max}$ = 3.4',ha='left',fontsize=8.5,color=PUR); ax.text(1.43,1.3,'$x_2$ enters at\n$\\alpha$ = 1.4',ha='left',fontsize=8.5,color=PUR,va='center')
    ax.plot([1.0,1.0],[10/9,1/9],'o',color=GREEN,ms=5,zorder=6); ax.text(0.93,0.55,'$\\alpha=1$:\n(10/9, 1/9)',fontsize=8.5,color=GREEN,ha='right',va='center')
    ax.invert_xaxis(); ax.set_xlabel('penalty $\\alpha$  (decreasing to the right)'); ax.set_ylabel('weight'); grid(ax); ax.legend(fontsize=8.5,loc='upper left',borderpad=0.2); ax.set_ylim(-0.08,1.5)
    fig.tight_layout(); save(fig,'f1151.png')

# ---- diabetes data shared by several figures
d=load_diabetes(); Xd,yd=d.data,d.target; names=list(d.feature_names)
Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.25,random_state=0)
cv=LassoCV(cv=5,random_state=0).fit(Xtr,ytr)
a_cv=float(cv.alpha_); amean=cv.mse_path_.mean(1); asem=cv.mse_path_.std(1,ddof=1)/np.sqrt(5); kb=amean.argmin(); a_1se=float(cv.alphas_[amean<=amean[kb]+asem[kb]][0])
def spread(ys,gap):
    o=np.argsort(ys); out=np.array(ys,float).copy(); pos=out[o]
    for i in range(1,len(pos)):
        if pos[i]-pos[i-1]<gap: pos[i]=pos[i-1]+gap
    # re-centre
    shift=(np.mean(out[o])-np.mean(pos)); pos=pos+shift; out[o]=pos; return out

# ---- 11.6-1 flow chart
if want('f1161'):
    fig,ax=plt.subplots(figsize=(4.6,4.5),dpi=240); ax.axis('off'); ax.set_xlim(0,4.6); ax.set_ylim(0,4.5)
    def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.6): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.05)
    A=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
    xc=1.95
    fb(xc,4.22,3.5,0.36,'Input: $\\mathbf{X}$, $\\mathbf{y}$, $\\alpha$, $\\rho$ (l1_ratio)',fc=LGREY,ec=GREY)
    fb(xc,3.55,3.5,0.5,'Centre if fit_intercept:  $\\mathbf{X}_c=\\mathbf{X}-\\bar{\\mathbf{x}}$,  $\\mathbf{y}_c=\\mathbf{y}-\\bar y$\nthen $v_j=\\frac{1}{n}\\mathbf{x}_j^{\\mathrm{T}}\\mathbf{x}_j$ for every column')
    fb(xc,2.95,3.5,0.36,'Start at $\\mathbf{w}=\\mathbf{0}$ (or the old $\\mathbf{w}$ if warm_start)',fc='#F1EAF7',ec=PUR)
    fb(xc,2.12,3.5,1.0,'One sweep: for each feature $j$ (in order,\nor random if selection="random")\n$z_j=\\frac{1}{n}\\mathbf{x}_j^{\\mathrm{T}}\\mathbf{r}+v_j w_j$\n$w_j\\leftarrow S(z_j,\\alpha\\rho)\\,/\\,(v_j+\\alpha(1-\\rho))$,  update $\\mathbf{r}$',fc='#E3F4EA',ec=GREEN,fs=8.4)
    ax.add_patch(Polygon([(xc,1.47),(xc+1.75,1.05),(xc,0.63),(xc-1.75,1.05)],fc='#FDF0E0',ec=OR,lw=0.9))
    ax.text(xc,1.05,'largest change $\\leq$ tol$\\cdot\\max|w_j|$\nand dual gap $\\leq$ tol$\\cdot\\|\\mathbf{y}\\|^2/n$ ?',ha='center',va='center',fontsize=8.3,color=NAVY)
    fb(xc,0.28,3.5,0.44,'$b=\\bar y-\\bar{\\mathbf{x}}^{\\mathrm{T}}\\mathbf{w}$;  store coef_, intercept_,\nn_iter_, dual_gap_',fc=LGREY,ec=GREY)
    A((xc,4.04),(xc,3.8)); A((xc,3.3),(xc,3.13)); A((xc,2.77),(xc,2.62)); A((xc,1.62),(xc,1.47)); A((xc,0.63),(xc,0.5))
    ax.text(xc+0.1,0.55,'yes',fontsize=8.5,color=GREEN,ha='left')
    ax.plot([xc+1.75,4.2,4.2],[1.05,1.05,2.12],color=NAVY,lw=0.9); A((4.2,2.12),(xc+1.75,2.12)); ax.text(4.25,1.6,'no:\nnext\nsweep',fontsize=8.5,color=RED,ha='left',va='center')
    ax.text(4.23,3.05,'at most\nmax_iter\nsweeps,\nthen\nConvergence\nWarning',fontsize=7.6,color=GREY,ha='left',va='center') if False else None
    fig.tight_layout(pad=0.1); save(fig,'f1161.png')

# ---- 11.9-1 gallery: coefficients as alpha shrinks
if want('f1191'):
    als=[2.0,0.5,0.1,0.01]; fig,axs=plt.subplots(1,4,figsize=(4.9,2.35),dpi=240,sharey=True); NUM['gallery']={}
    for ax,a in zip(axs,als):
        m=Lasso(alpha=a,max_iter=100000,tol=1e-8).fit(Xtr,ytr); w=m.coef_; nz=int((w!=0).sum())
        NUM['gallery'][str(a)]={'nonzero':nz,'test_r2':round(float(m.score(Xte,yte)),3),'coef':[round(float(v),1) for v in w]}
        y_=np.arange(10)[::-1]
        ax.barh(y_,w,color=GREEN,height=0.62); ax.plot(np.zeros((w==0).sum()),y_[w==0],'|',color=GREY,ms=7)
        ax.axvline(0,color=GREY,lw=0.6); ax.set_xlim(-300,720); ax.set_xticks([0,500]); ax.tick_params(axis='x',labelsize=8.5)
        ax.set_title(f'$\\alpha$ = {a:g}\n{nz} non-zero',fontsize=9,color=PUR); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
    axs[0].set_yticks(np.arange(10)[::-1]); axs[0].set_yticklabels(names,fontsize=8.8)
    fig.tight_layout(w_pad=0.3,rect=(0,0.06,1,1)); fig.supxlabel('weight $w_j$ of the lasso fit on the diabetes training data',fontsize=9,y=0.0)
    save(fig,'f1191.png')

# ---- 11.9-2 coefficient path
if want('f1192'):
    grid_a=np.geomspace(2.4,2.4e-3,200); a_,c_,_=lasso_path(Xtr-Xtr.mean(0),ytr-ytr.mean(),alphas=grid_a,tol=1e-9,max_iter=100000)
    nzp=(c_!=0).sum(0); NUM['path']={'a_max':float(a_[0]),'nz_at_end':int(nzp[-1])}
    ent=[float(a_[np.argmax(c_[j]!=0)]) if (c_[j]!=0).any() else None for j in range(10)]; NUM['path']['entry_alpha']={names[j]:(round(ent[j],3) if ent[j] else None) for j in range(10)}
    fig,ax=plt.subplots(figsize=(4.9,2.9),dpi=240)
    sty={'bmi':(BLUE,'-'),'s5':(GREEN,'-'),'bp':(OR,'-'),'s3':(PUR,'-'),'sex':(RED,'-'),'s1':(MAG,'--'),'s2':(TEAL,'--')}
    for j,nm in enumerate(names):
        col,ls=sty.get(nm,(GREY,':')); ax.plot(a_,c_[j],color=col,ls=ls,lw=1.6 if nm in sty else 1.0)
    endy=c_[:,-1]; ly=spread(endy,48.0)
    for j,nm in enumerate(names):
        col,ls=sty.get(nm,(GREY,':')); ax.text(a_[-1]*0.8,ly[j],nm,fontsize=8.6,color=col,ha='left',va='center')
    ax.axvline(a_cv,color=PUR,lw=0.9,ls='--'); ax.text(a_cv*1.1,-440,f'CV choice\n$\\alpha$ = {a_cv:.3f}',fontsize=8.5,color=PUR,va='center',ha='right')
    ax.set_xscale('log'); ax.set_xlabel('penalty $\\alpha$ (log scale, decreasing to the right)'); ax.set_ylabel('weight $w_j$'); grid(ax)
    ax.set_xlim(a_[0]*1.15,a_[-1]*0.42)
    fig.tight_layout(); save(fig,'f1192.png')

# ---- 11.9-3 elastic-net paths for three l1_ratio
if want('f1193'):
    fig,axs=plt.subplots(1,3,figsize=(4.9,2.2),dpi=240,sharey=True); grid_a=np.geomspace(30,3e-4,160); NUM['enet_paths']={}
    Xc_,yc_=Xtr-Xtr.mean(0),ytr-ytr.mean()
    for ax,rho in zip(axs,[1.0,0.5,0.1]):
        a_,c_,_=enet_path(Xc_,yc_,l1_ratio=rho,alphas=grid_a,tol=1e-9,max_iter=100000)
        for j,nm in enumerate(names):
            if nm=='s1': ax.plot(a_,c_[j],color=MAG,ls='--',lw=1.6,label='s1')
            elif nm=='s2': ax.plot(a_,c_[j],color=TEAL,ls='-.',lw=1.6,label='s2')
            else: ax.plot(a_,c_[j],color=GREY,lw=0.8,alpha=0.8,label='other 8' if nm=='age' else None)
        ax.set_xscale('log'); ax.set_xlim(30,3e-4); ax.set_xticks([10,0.1,0.001]); ax.set_xticklabels(['10','0.1','0.001'],fontsize=8.5)
        ax.set_title(f'$\\rho$ = {rho:g}'+(' (lasso)' if rho==1 else ''),fontsize=9,color=PUR); grid(ax)
        ax.set_xlabel('$\\alpha$',fontsize=9)
        i0=int(np.argmax((c_!=0).any(0))); NUM['enet_paths'][str(rho)]={'alpha_first_nonzero':round(float(a_[i0]),3)}
    axs[0].set_ylabel('weight $w_j$'); axs[1].legend(*axs[0].get_legend_handles_labels(),fontsize=8.2,loc='lower left',borderpad=0.2,labelspacing=0.2,handlelength=1.6)
    fig.tight_layout(w_pad=0.4); save(fig,'f1193.png')

# ---- 11.9-4 validation curve
if want('f1194'):
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.3),dpi=240); al=cv.alphas_
    ax=axs[0]; ax.fill_between(al,amean-asem,amean+asem,color='#F1EAF7'); ax.plot(al,amean,color=PUR,lw=1.6)
    ax.plot(a_cv,amean[kb],'o',color=GREEN,ms=6,mec='white',zorder=6); ax.plot(a_1se,amean[al==a_1se][0],'s',color=OR,ms=6,mec='white',zorder=6)
    ax.text(a_cv*0.9,amean[kb]+260,'minimum',fontsize=8.5,color=GREEN,ha='left'); ax.text(a_1se*1.1,amean[al==a_1se][0]+260,'one-s.e. rule',fontsize=8.5,color=OR,ha='left')
    ax.set_xscale('log'); ax.set_xlabel('$\\alpha$'); ax.set_ylabel('cross-validated MSE'); grid(ax); ax.set_ylim(2850,4300); ax.set_xlim(al[0]*1.2,al[-1]*0.8)
    ax=axs[1]; nzs=[]
    for a in al: nzs.append(int((Lasso(alpha=a,max_iter=100000,tol=1e-7,warm_start=False).fit(Xtr,ytr).coef_!=0).sum()))
    ax.step(al,nzs,where='post',color=GREEN,lw=1.6); ax.axvline(a_cv,color=GREEN,lw=0.8,ls=':'); ax.axvline(a_1se,color=OR,lw=0.8,ls=':')
    ax.set_xscale('log'); ax.set_xlabel('$\\alpha$'); ax.set_ylabel('non-zero weights'); grid(ax); ax.set_yticks([0,2,4,6,8,10]); ax.set_xlim(al[0]*1.2,al[-1]*0.8)
    NUM['cv']={'alpha_cv':round(a_cv,4),'alpha_1se':round(a_1se,4),'min_mse':round(float(amean[kb]),1),'mse_1se':round(float(amean[al==a_1se][0]),1),'sem_at_min':round(float(asem[kb]),1),'nz_cv':int(nzs[kb]),'nz_1se':int(nzs[int(np.argmax(al==a_1se))])}
    fig.tight_layout(w_pad=0.9); save(fig,'f1194.png')

# ---- 11.10-1 LARS geometry and equal correlations
if want('f1101'):
    from matplotlib.patches import Arc
    x1=Xtc[:,0]; x2=Xtc[:,1]; e1=x1/np.linalg.norm(x1); r2=x2-(x2@e1)*e1; e2=r2/np.linalg.norm(r2)
    P=lambda v: np.array([v@e1,v@e2])
    p1,p2=P(x1),P(x2); yhat=P(Xtc@what); mu1=P(1.0*x1); u=(p1+p2)/np.linalg.norm(p1+p2)
    th0=np.arctan2(u[1],u[0]); R=np.array([[np.cos(-th0),-np.sin(-th0)],[np.sin(-th0),np.cos(-th0)]])   # rotate so that the equal-angle line is horizontal
    p1,p2,yhat,mu1,u=[R@v for v in (p1,p2,yhat,mu1,u)]
    ang1=np.degrees(np.arctan2(p1[1],p1[0])); ang2=np.degrees(np.arctan2(p2[1],p2[0]))
    NUM['lars_geom']={'angle_x1_x2_deg':round(ang2-ang1,1),'half_angle_deg':round(ang2,1),'step2_length':round(float(np.linalg.norm(yhat-mu1)),3),'norm_x1_plus_x2':round(float(np.linalg.norm(x1+x2)),3)}
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.4),dpi=240,gridspec_kw={'width_ratios':[1.2,1]})
    ax=axs[0]; ax.set_aspect('equal'); ax.axis('off')
    ax.plot([0,6.6],[0,0],color=GREY,lw=0.8,ls=':'); ax.text(6.7,0.12,'equal-angle line',fontsize=8.4,color=GREY,ha='right',va='bottom')
    ax.plot([0,mu1[0]],[0,mu1[1]],color=GREEN,lw=3.2,alpha=0.55,solid_capstyle='butt'); ax.plot([mu1[0],yhat[0]],[mu1[1],yhat[1]],color=GREEN,lw=3.2,alpha=0.55,solid_capstyle='butt')
    L=2.1; arr(ax,(0,0),tuple(L*p1/np.linalg.norm(p1)),c=BLUE,lw=1.3,ms=9); arr(ax,(0,0),tuple(L*p2/np.linalg.norm(p2)),c=OR,lw=1.3,ms=9)
    ax.plot(*mu1,'o',color=GREEN,ms=6,mec='white',zorder=6); ax.plot(*yhat,'*',color=GREEN,ms=11,mec='white',zorder=6)
    ax.add_patch(Arc((0,0),3.0,3.0,theta1=ang1,theta2=0,color=PUR,lw=1.0)); ax.add_patch(Arc((0,0),3.0,3.0,theta1=0,theta2=ang2,color=PUR,lw=1.0))
    ax.text(1.9,-0.3,'$\\theta$',color=PUR,fontsize=9.5,ha='center',va='center'); ax.text(1.9,0.3,'$\\theta$',color=PUR,fontsize=9.5,ha='center',va='center')
    ax.text(L*np.cos(np.radians(ang2))-0.1,L*np.sin(np.radians(ang2))+0.25,'$\\mathbf{x}_2$',color=OR,fontsize=10,ha='center'); ax.text(1.0,-0.72,'$\\mathbf{x}_1$',color=BLUE,fontsize=10,ha='center',va='top')
    cb=dict(boxstyle='circle,pad=0.18',fc='white',ec=GREEN,lw=0.9)
    ax.text(mu1[0]*0.55+0.2,mu1[1]*0.55-0.62,'1',color=GREEN,fontsize=9,ha='center',va='center',bbox=cb); ax.text((mu1[0]+yhat[0])/2,mu1[1]+0.5,'2',color=GREEN,fontsize=9,ha='center',va='center',bbox=cb)
    ax.text(yhat[0]+0.3,yhat[1]-0.3,'$\\hat{\\mathbf{y}}$ (least squares)',color=GREEN,fontsize=8.4,ha='center',va='top')
    ax.set_xlim(-0.3,6.8); ax.set_ylim(-2.2,1.6)
    ax=axs[1]
    al_=np.linspace(3.4,0.0,171); a_,c_,_=lasso_path(Xtc,ytc,alphas=al_[:-1].tolist()+[1e-6],tol=1e-12,max_iter=100000)
    cor=np.array([[abs(Xtc[:,j]@(ytc-Xtc@c_[:,k]))/n5 for k in range(c_.shape[1])] for j in range(2)])
    ax.plot(a_,a_,color=GRID,lw=3.5,label='$|c| = \\alpha$'); ax.plot(a_,cor[0],color=BLUE,lw=1.5,label='$|c_1|$'); ax.plot(a_,cor[1],color=OR,lw=1.5,ls='--',label='$|c_2|$')
    ax.axvline(1.4,color=PUR,lw=0.8,ls=':'); ax.text(1.37,0.15,'tie',color=PUR,fontsize=8.5,ha='left'); ax.invert_xaxis()
    ax.set_xlabel('$\\alpha$ (decreasing)'); ax.set_ylabel('correlation with the residual'); grid(ax); ax.legend(fontsize=8.5,loc='upper right',borderpad=0.2,labelspacing=0.2)
    fig.tight_layout(w_pad=0.6); save(fig,'f1101.png')

# ---- 11.11-1 OMP: scores at each step
if want('f11111'):
    Xs_,ys_,wt_=make_regression(n_samples=120,n_features=20,n_informative=4,noise=25.0,coef=True,random_state=2)
    Xc_=Xs_-Xs_.mean(0); yc_=ys_-ys_.mean(); chosen=[]; r=yc_.copy(); fig,axs=plt.subplots(1,4,figsize=(4.9,2.1),dpi=240); true=np.flatnonzero(wt_); NUM['omp']={'steps':[]}
    for k,ax in enumerate(axs):
        sc=np.abs(Xc_.T@r)/120; sc2=sc.copy(); sc2[chosen]=0; j=int(sc2.argmax())
        cols=[GREEN if i==j else (GREY if i in chosen else BLUE) for i in range(20)]
        ax.bar(np.arange(20),sc,color=cols,width=0.75); ax.set_title(f'step {k+1}: pick {j}',fontsize=9,color=NAVY)
        ax.plot(true,np.full(4,-0.07*sc.max()),'^',color=NAVY,ms=3.5,clip_on=False)
        chosen.append(j); w=np.linalg.lstsq(Xc_[:,chosen],yc_,rcond=None)[0]; r=yc_-Xc_[:,chosen]@w
        NUM['omp']['steps'].append({'pick':j,'score':round(float(sc2[j]),2),'rss':round(float(r@r),0)})
        ax.set_xticks([0,10,19]); ax.tick_params(labelsize=8.5); ax.set_xlabel('feature $j$',fontsize=9); ax.set_ylim(0,None); grid(ax)
    axs[0].set_ylabel('score $|\\mathbf{x}_j^{\\mathrm{T}}\\mathbf{r}|/n$',fontsize=9)
    fig.tight_layout(w_pad=0.6); save(fig,'f11111.png')

# ---- 11.12-1 multi-task: row sparsity
if want('f11121'):
    rng=np.random.default_rng(11); n_,p_,K_=30,12,4; X_=rng.normal(size=(n_,p_)); W_=np.zeros((p_,K_)); rows=[1,4,8]
    W_[rows]=rng.choice([-1,1],size=(3,K_))*rng.uniform(0.6,1.6,size=(3,K_)); Y_=X_@W_+rng.normal(0,2.0,size=(n_,K_)); a=0.95
    mt=MultiTaskLasso(alpha=a,tol=1e-9,max_iter=100000).fit(X_,Y_).coef_.T
    sep=np.array([Lasso(alpha=a,tol=1e-9,max_iter=100000).fit(X_,Y_[:,k]).coef_ for k in range(K_)]).T
    truth=(W_!=0); NUM['multitask']={'alpha':a,'mt_rows':[int(v) for v in np.flatnonzero((mt!=0).any(1))],'sep_missed':int((truth&(sep==0)).sum()),'sep_false':int((~truth&(sep!=0)).sum()),'sep_nnz':int((sep!=0).sum()),'mt_nnz':int((mt!=0).sum()),'true_nnz':int(truth.sum())}
    fig,axs=plt.subplots(1,3,figsize=(4.9,2.5),dpi=240,sharey=True)
    for ax,M,t_ in zip(axs,[W_,sep,mt],['true weights','separate lassos','MultiTaskLasso']):
        im=ax.imshow(M,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto',interpolation='nearest')
        ax.set_xticks(range(K_)); ax.set_xticklabels([f'{k+1}' for k in range(K_)],fontsize=8.5); ax.set_xlabel('task $k$',fontsize=9); ax.set_title(t_,fontsize=9,color=NAVY)
        for i in range(p_+1): ax.axhline(i-0.5,color='white',lw=0.4)
        for kk in range(K_+1): ax.axvline(kk-0.5,color='white',lw=0.4)
        for i,j in zip(*np.nonzero(M)): ax.plot(j,i,'.',color='k',ms=2.2)
    axs[0].set_yticks(range(p_)); axs[0].set_yticklabels([str(i) for i in range(p_)],fontsize=8.2); axs[0].set_ylabel('feature $j$',fontsize=9)
    fig.tight_layout(w_pad=0.5); save(fig,'f11121.png')

# ---- 11.14-1 case study
if want('f11141'):
    ols=LinearRegression().fit(Xtr,ytr); en=ElasticNetCV(l1_ratio=0.5,eps=1e-4,cv=5,random_state=0).fit(Xtr,ytr)
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.6),dpi=240,gridspec_kw={'width_ratios':[1.05,1]})
    ax=axs[0]; y_=np.arange(10)[::-1]
    ax.axvline(0,color=GREY,lw=0.6)
    ax.scatter(ols.coef_,y_+0.2,marker='o',s=24,color=BLUE,label='least squares',zorder=4)
    ax.scatter(cv.coef_,y_,marker='^',s=30,color=GREEN,label='LassoCV',zorder=5)
    ax.scatter(en.coef_,y_-0.2,marker='s',s=22,color=OR,label='ElasticNetCV',zorder=4)
    ax.set_yticks(y_); ax.set_yticklabels(names,fontsize=8.8); ax.set_xlabel('weight $w_j$'); ax.grid(axis='y',color=GRID,lw=0.4); ax.set_axisbelow(True)
    ax.legend(fontsize=8.2,loc='lower left',borderpad=0.2,labelspacing=0.2,handletextpad=0.2); ax.set_xlim(-1000,800)
    ax=axs[1]; al=np.geomspace(2.4,1e-3,120); r2=[];nz=[]
    for a in al:
        m=Lasso(alpha=a,max_iter=100000,tol=1e-8).fit(Xtr,ytr); r2.append(m.score(Xte,yte)); nz.append(int((m.coef_!=0).sum()))
    r2=np.array(r2); nz=np.array(nz); best={}
    for k in range(0,11):
        idx=np.flatnonzero(nz==k)
        if len(idx): best[k]=float(r2[idx].max())
    ks=sorted(best); ax.plot(ks,[best[k] for k in ks],'-o',color=GREEN,ms=4,lw=1.4,label='lasso, best test $R^2$ for each size')
    ax.axhline(ols.score(Xte,yte),color=BLUE,lw=1.0,ls='--'); ax.text(0.1,ols.score(Xte,yte)+0.012,'least squares 0.359',color=BLUE,fontsize=8.5)
    ax.set_xlabel('number of non-zero weights'); ax.set_ylabel('test $R^2$'); grid(ax); ax.set_xticks([0,2,4,6,8,10]); ax.set_ylim(-0.03,0.42)
    NUM['case']={'r2_by_size':{str(k):round(best[k],3) for k in ks},'ols_test':round(float(ols.score(Xte,yte)),3)}
    fig.tight_layout(w_pad=0.8); save(fig,'f11141.png')

# ---- 11.10-2 information criteria
if want('f1102'):
    Xs_,ys_,wt_=make_regression(n_samples=120,n_features=20,n_informative=4,noise=25.0,coef=True,random_state=2)
    fig,ax=plt.subplots(figsize=(3.6,2.2),dpi=240); NUM['ic']={}
    for crit,col,mk,ls in [('aic',OR,'s','--'),('bic',GREEN,'o','-')]:
        ic=LassoLarsIC(criterion=crit).fit(Xs_,ys_); nzc=np.arange(len(ic.criterion_))
        cc=ic.criterion_-ic.criterion_.min(); ax.plot(nzc,cc,color=col,marker=mk,ms=3.5,lw=1.3,ls=ls,label=crit.upper()); b=int(np.argmin(ic.criterion_)); ax.plot(b,0,marker=mk,ms=9,mfc='none',mec=col,mew=1.4)
        NUM['ic'][crit]={'best_node':b,'alpha':round(float(ic.alpha_),3),'sel':[int(v) for v in np.flatnonzero(ic.coef_)]}
    ax.text(0.2,55,'nodes 0 to 2\nare off scale',fontsize=8.3,color=GREY,va='center'); ax.set_xlabel('node of the path (0 = all weights zero)'); ax.set_ylabel('criterion $-$ its minimum'); grid(ax); ax.legend(fontsize=8.5,loc='upper right',borderpad=0.2); ax.set_xlim(-0.5,20.5); ax.set_ylim(-8,95)
    fig.tight_layout(); save(fig,'f1102.png')

with open('verify/fig_numbers.json','w') as fh: json.dump(NUM,fh,indent=1)

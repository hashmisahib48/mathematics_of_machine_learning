# All main figures of Chapter 18. Run: python3 fig/figs.py   (PNGs into ch18/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
C3=[BLUE,OR,TEAL]; M3=['o','s','^']
LC=['#BBD3F2','#FBD9A8','#BFE3E6']        # light class fills
def ell(ax,mu,S,k=2.0,**kw):
    ev,U=np.linalg.eigh(S); ang=np.degrees(np.arctan2(U[1,1],U[0,1]))
    ax.add_patch(Ellipse(mu,2*k*np.sqrt(ev[1]),2*k*np.sqrt(ev[0]),angle=ang,fill=False,**kw))
def grid(ax,pad=0.3,n=300):
    x0,x1=ax.get_xlim(); y0,y1=ax.get_ylim(); gx,gy=np.meshgrid(np.linspace(x0,x1,n),np.linspace(y0,y1,n)); return gx,gy,np.c_[gx.ravel(),gy.ravel()]
def dots(ax,X,y,s=16,cls=(0,1),alpha=0.9):
    for k in cls: ax.scatter(X[y==k,0],X[y==k,1],s=s,color=C3[k],marker=M3[k],edgecolor='white',lw=0.5,zorder=4,alpha=alpha)

# ---------------- f180 opener: LDA line and QDA curve on two Gaussian classes
rng=np.random.default_rng(4)
S0=np.array([[0.5,0.2],[0.2,0.4]]); S1=np.array([[3.2,0.6],[0.6,3.0]]); m0=np.array([0.,0.]); m1=np.array([1.6,0.6])
X0=rng.multivariate_normal(m0,S0,70); X1=rng.multivariate_normal(m1,S1,70); X=np.r_[X0,X1]; y=np.r_[np.zeros(70),np.ones(70)].astype(int)
lda=LDA().fit(X,y); qda=QDA().fit(X,y); Xt=np.r_[rng.multivariate_normal(m0,S0,2000),rng.multivariate_normal(m1,S1,2000)]; yt=np.r_[np.zeros(2000),np.ones(2000)].astype(int)
NUM['opener']={'lda_test':round(lda.score(Xt,yt),4),'qda_test':round(qda.score(Xt,yt),4)}
fig,ax=plt.subplots(figsize=(4.6,2.9),dpi=240); ax.set_xlim(-4.6,6.6); ax.set_ylim(-4.2,4.6)
dots(ax,X,y,s=18)
for mu,S,c in ((lda.means_[0],np.cov(X0.T,bias=True),BLUE),(lda.means_[1],np.cov(X1.T,bias=True),OR)): ell(ax,mu,S,2.0,ec=c,lw=1.0,ls=':')
gx,gy,G=grid(ax)
ax.contour(gx,gy,lda.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=2.0,linestyles='-')
ax.contour(gx,gy,qda.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=2.0,linestyles='--')
ax.legend(handles=[Line2D([],[],color=GREEN,lw=2,ls='-',label='LDA boundary (straight)'),Line2D([],[],color=GREEN,lw=2,ls='--',label='QDA boundary (curved)'),
   Line2D([],[],color=BLUE,marker='o',ls='',ms=4,label='class 0'),Line2D([],[],color=OR,marker='s',ls='',ms=4,label='class 1')],loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize=8.5,ncol=2,columnspacing=1.2,handlelength=1.8)
ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
sv(fig,'f180.png',420)
# ---------------- f181 seeds: densities, priors, posterior
A=np.array([1.,3,3,3,3,5]); Bv=np.array([6.,7,7,8]); xs=np.linspace(0,10,600)
m0_,m1_=3.0,7.0; v_=1.0; pi0_,pi1_=0.6,0.4; v0_,v1_=A.var(),Bv.var()
npdf=lambda x,m,v:np.exp(-(x-m)**2/(2*v))/np.sqrt(2*np.pi*v)
fig,(a1,a2)=plt.subplots(2,1,figsize=(4.7,3.7),dpi=240,sharex=True,gridspec_kw={'height_ratios':[1.15,1],'hspace':0.12})
a1.plot(xs,pi0_*npdf(xs,m0_,v_),color=BLUE,lw=1.8); a1.plot(xs,pi1_*npdf(xs,m1_,v_),color=OR,lw=1.8,ls='--')
a1.fill_between(xs,0,pi0_*npdf(xs,m0_,v_),color=LC[0],alpha=0.6); a1.fill_between(xs,0,pi1_*npdf(xs,m1_,v_),color=LC[1],alpha=0.6)
for vals,c,mk in ((A,BLUE,'o'),(Bv,OR,'s')):
    for u in np.unique(vals):
        k=np.sum(vals==u); a1.scatter([u]*k,0.012+0.032*np.arange(k),s=22,color=c,marker=mk,edgecolor='white',lw=0.5,zorder=5)
a1.text(1.6,0.2,'variety A\n$\\pi_0\\, p(x\\mid A)$',color=BLUE,fontsize=9,ha='center'); a1.text(8.5,0.19,'variety B\n$\\pi_1\\, p(x\\mid B)$',color=OR,fontsize=9,ha='center')
a1.set_ylabel('density × prior'); a1.set_ylim(0,0.30); a1.grid(color=GRID,lw=0.5); a1.set_axisbelow(True)
zl=lambda x,v0,v1,m0,m1:(-0.5*(x-m1)**2/v1-0.5*np.log(v1)+np.log(pi1_))-(-0.5*(x-m0)**2/v0-0.5*np.log(v0)+np.log(pi0_))
sg=lambda z:1/(1+np.exp(-z))
a2.plot(xs,sg(zl(xs,v_,v_,m0_,m1_)),color=GREEN,lw=2.0,label='LDA, priors 0.6 / 0.4')
pe=sg(zl(xs,v_,v_,m0_,m1_)-np.log(pi1_/pi0_)); a2.plot(xs,pe,color=GREY,lw=1.6,ls=':',label='LDA, equal priors')
a2.plot(xs,sg(zl(xs,v0_,v1_,m0_,m1_)),color=GREEN,lw=1.6,ls='--',label='QDA')
a2.axhline(0.5,color=GRID,lw=0.9,zorder=0); a2.axvline(5.1014,color=GREEN,lw=0.8,ls='-',alpha=0.6)
a2.set_ylabel('$P(\\mathrm{B}\\mid x)$'); a2.set_xlabel('seed length $x$ (mm)'); a2.set_xlim(0,10); a2.set_ylim(-0.04,1.04); a2.grid(color=GRID,lw=0.5); a2.set_axisbelow(True)
a2.legend(loc='upper left',fontsize=8.5,handlelength=2.2)
sv(fig,'f181.png',450)
# ---------------- f182 whitening: LDA = nearest centroid in Mahalanobis distance
rng=np.random.default_rng(11); Sg=np.array([[1.0,0.8],[0.8,1.0]]); mu0=np.array([-1.3,-0.3]); mu1=np.array([1.3,0.7])
Xa=np.r_[rng.multivariate_normal(mu0,Sg,60),rng.multivariate_normal(mu1,Sg,60)]; ya=np.r_[np.zeros(60),np.ones(60)].astype(int)
la=LDA(store_covariance=True).fit(Xa,ya); Sh=la.covariance_; ev,U=np.linalg.eigh(Sh); Wh=U@np.diag(ev**-0.5)@U.T
xb=la.xbar_; Zw=(Xa-xb)@Wh; mz=(la.means_-xb)@Wh
NUM['whiten']={'cov_within_after':np.round(np.cov(np.r_[Zw[ya==0]-mz[0],Zw[ya==1]-mz[1]].T,bias=True),3).tolist(),'maha':round(float(np.linalg.norm(mz[1]-mz[0])),4),'eucl':round(float(np.linalg.norm(la.means_[1]-la.means_[0])),4),'n':int(len(ya))}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.45),dpi=240)
ax=axs[0]; dots(ax,Xa,ya,s=12)
for k in (0,1): ell(ax,la.means_[k],Sh,2.0,ec=C3[k],lw=1.0,ls=':')
mid=(la.means_[0]+la.means_[1])/2; dd=la.means_[1]-la.means_[0]; t=np.array([-dd[1],dd[0]]); t/=np.linalg.norm(t)
ax.plot(*np.c_[mid-4*t,mid+4*t],color=GREY,lw=1.4,ls='--'); ax.set_xlim(-4.6,4.6); ax.set_ylim(-3.8,3.8); ax.set_aspect('equal')
gx,gy,G=grid(ax); ax.contour(gx,gy,la.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=2.0)
ax.set_title('raw features\ngreen: LDA;  grey: nearest centroid',fontsize=8.8,color=NAVY)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
ax=axs[1]; dots(ax,Zw,ya,s=12)
for k in (0,1): ax.add_patch(Circle(mz[k],2.0,fill=False,ec=C3[k],lw=1.0,ls=':'))
mz_mid=(mz[0]+mz[1])/2; dz=mz[1]-mz[0]; tz=np.array([-dz[1],dz[0]]); tz/=np.linalg.norm(tz)
ax.plot(*np.c_[mz_mid-5*tz,mz_mid+5*tz],color=GREEN,lw=2.0); ax.plot(*np.c_[mz[0],mz[1]],color=NAVY,lw=1.0,marker='+',ms=7)
ax.set_xlim(-4.6,4.6); ax.set_ylim(-3.8,3.8); ax.set_aspect('equal'); ax.set_title('whitened features\nLDA = perpendicular bisector',fontsize=8.8,color=NAVY); ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
for ax in axs: ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f182.png',450)
# ---------------- f183 Fisher's criterion: two directions
from matplotlib.gridspec import GridSpec
Sw_=Sh; dmu=la.means_[1]-la.means_[0]; aF=np.linalg.solve(Sw_,dmu); aF/=np.linalg.norm(aF); aM=dmu/np.linalg.norm(dmu)
Jf=lambda a:float((a@dmu)**2/(a@Sw_@a)); NUM['fisher']={'J_meandiff':round(Jf(aM),3),'J_fisher':round(Jf(aF),3),'maha2':round(float(dmu@np.linalg.solve(Sw_,dmu)),3),'angle_deg':round(float(np.degrees(np.arccos(abs(aM@aF)))),1)}
fig=plt.figure(figsize=(4.9,2.55),dpi=240); gs=GridSpec(2,2,width_ratios=[1.15,1],hspace=0.55,wspace=0.32,figure=fig)
ax=fig.add_subplot(gs[:,0]); dots(ax,Xa,ya,s=11); c0=xb
for a_,c,ls in ((aM,GREY,'--'),(aF,GREEN,'-')): ax.plot(*np.c_[c0-4.6*a_,c0+4.6*a_],color=c,lw=1.8,ls=ls,zorder=3)
ax.set_xlim(-4.6,4.6); ax.set_ylim(-3.8,3.8); ax.set_aspect('equal'); ax.set_xticks([-4,-2,0,2,4]); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.text(3.6,-1.55,'$\\mathbf{a}_{\\mathrm{F}}$',color=GREEN,fontsize=10,ha='center'); ax.text(-1.5,-3.05,'$\\mu_1-\\mu_0$ direction',color=GREY,fontsize=8.5,ha='left')
bins=np.linspace(-7,7,29)
for r,(a_,c,t) in enumerate(((aM,GREY,'along $\\mu_1-\\mu_0$'),(aF,GREEN,'along Fisher direction'))):
    ax=fig.add_subplot(gs[r,1]); pr=(Xa-c0)@a_
    for k in (0,1): ax.hist(pr[ya==k],bins=bins,color=LC[k],ec=C3[k],lw=0.7,alpha=0.85,histtype='stepfilled')
    ax.set_title(f'{t}:  $J$ = {Jf(a_):.1f}',fontsize=8.8,color=c if c!=GREY else NAVY); ax.set_yticks([]); ax.set_xlim(-7,7); ax.set_xticks([-4,0,4]); ax.spines['left'].set_visible(False); ax.grid(axis='x',color=GRID,lw=0.5)
sv(fig,'f183.png',450)
# ---------------- f184 wine: LDA against PCA
Xw,yw=load_wine(return_X_y=True); lw=LDA().fit(Xw,yw); Zl=lw.transform(Xw); Zs=StandardScaler().fit_transform(Xw); pc=PCA(2).fit(Zs); Zp=pc.transform(Zs)
NUM['wine']={'lda_ratio':np.round(lw.explained_variance_ratio_,4).tolist(),'pca_ratio':np.round(pc.explained_variance_ratio_,4).tolist(),
  'lda_train_acc_2d':round(float(LDA().fit(Zl,yw).score(Zl,yw)),4),'pca_train_acc_2d':round(float(LDA().fit(Zp,yw).score(Zp,yw)),4),'lda_train_acc_full':round(float(lw.score(Xw,yw)),4)}
names=['class 0','class 1','class 2']
fig,axs=plt.subplots(1,2,figsize=(4.9,2.5),dpi=240)
for ax,Z_,xl,yl,t in ((axs[0],Zl,'discriminant 1  (%.0f%%)'%(100*lw.explained_variance_ratio_[0]),'discriminant 2  (%.0f%%)'%(100*lw.explained_variance_ratio_[1]),'LDA: 13 features to 2'),
                      (axs[1],Zp,'principal component 1','principal component 2','PCA: 13 features to 2')):
    for k in range(3): ax.scatter(Z_[yw==k,0],Z_[yw==k,1],s=11,color=C3[k],marker=M3[k],edgecolor='white',lw=0.4,label=names[k],zorder=4)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t,fontsize=9,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
axs[0].legend(loc='lower right',fontsize=8,handletextpad=0.1,borderaxespad=0.2,labelspacing=0.2)
axs[0].set_ylim(-5.4,4.6); fig.tight_layout(w_pad=1.0); sv(fig,'f184.png',450)
# ---------------- f185 flow chart of fit
fig,ax=new(3.9,4.7)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.2): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.05)
A=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
fb(1.95,4.5,3.7,0.42,'Input:  $\\mathbf{X}$ ($n$ rows, $p$ columns)  and labels $y$',fc=LGREY,ec=GREY)
fb(1.95,3.82,3.7,0.5,'Priors $\\pi_k=n_k/n$ and class means $\\mu_k$\n(`priors` replaces the proportions)'.replace('`',''))
fb(1.95,3.02,3.7,0.62,'Shared covariance  $\\Sigma=\\sum_k \\pi_k\\hat\\Sigma_k$  (each divided by $n_k$)\noptional shrinkage, Eq. 18.5-3')
ax.add_patch(Polygon([(1.95,2.55),(2.95,2.2),(1.95,1.85),(0.95,2.2)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(1.95,2.2,'solver?',ha='center',va='center',fontsize=8.4,color=NAVY)
fb(0.68,1.27,1.12,0.95,'svd (default)\nwhiten by SVD\nof centred data;\nno $\\Sigma$ formed',fc='#E3F4EA',ec=GREEN,fs=7.8)
fb(1.95,1.27,1.12,0.95,'lsqr\nsolve\n$\\Sigma\\,\\mathbf{W}=\\mathbf{M}^{\\mathrm{T}}$\n(no transform)',fc='#F1EAF7',ec=PUR,fs=7.8)
fb(3.22,1.27,1.12,0.95,'eigen\n$S_b\\mathbf{a}=\\lambda S_w\\mathbf{a}$\n(also gives\ntransform)',fc='#F1EAF7',ec=PUR,fs=7.8)
fb(1.95,0.38,3.7,0.5,'$\\mathbf{w}_k=\\Sigma^{-1}\\mu_k$,   $b_k=-\\frac{1}{2}\\mu_k^{\\mathrm{T}}\\Sigma^{-1}\\mu_k+\\ln\\pi_k$   (Eq. 18.3-5)',fs=8.2)
fb(1.95,-0.32,3.7,0.4,'Store  coef_, intercept_, means_, priors_',fc=LGREY,ec=GREY)
for p_,q_ in (((1.95,4.29),(1.95,4.07)),((1.95,3.57),(1.95,3.33)),((1.95,2.71),(1.95,2.55))): A(p_,q_)
ax.plot([1.2,0.68],[2.2,2.2],color=NAVY,lw=0.9); A((0.68,2.2),(0.68,1.75)); A((1.95,1.85),(1.95,1.75)); ax.plot([2.7,3.22],[2.2,2.2],color=NAVY,lw=0.9); A((3.22,2.2),(3.22,1.75))
for xt in (0.68,1.95,3.22): A((xt,0.79),(xt,0.64))
A((1.95,0.13),(1.95,-0.12))
ax.set_xlim(0,3.9); ax.set_ylim(-0.6,4.8); sv(fig,'f185.png',370)
# ---------------- f186 gallery: shrinkage (LDA) and reg_param (QDA)
SH=[0.0,0.3,0.7,1.0]
rngg=np.random.default_rng(21); Sgl=np.array([[1.0,0.8],[0.8,1.0]]); ml0=np.array([0.,0.]); ml1=np.array([1.9,0.4])
XL=np.r_[rngg.multivariate_normal(ml0,Sgl,14),rngg.multivariate_normal(ml1,Sgl,14)]; yL=np.r_[np.zeros(14),np.ones(14)].astype(int)
XLt=np.r_[rngg.multivariate_normal(ml0,Sgl,5000),rngg.multivariate_normal(ml1,Sgl,5000)]; yLt=np.r_[np.zeros(5000),np.ones(5000)].astype(int)
Sq0=np.array([[1.4,0.9],[0.9,1.0]]); Sq1=np.array([[1.0,-0.7],[-0.7,1.6]]); mq0=np.array([0.,0.]); mq1=np.array([1.8,0.8])
XQ=np.r_[rngg.multivariate_normal(mq0,Sq0,16),rngg.multivariate_normal(mq1,Sq1,16)]; yQ=np.r_[np.zeros(16),np.ones(16)].astype(int)
XQt=np.r_[rngg.multivariate_normal(mq0,Sq0,5000),rngg.multivariate_normal(mq1,Sq1,5000)]; yQt=np.r_[np.zeros(5000),np.ones(5000)].astype(int)
NUM['gallery']={'lda':{},'qda':{}}
fig,axs=plt.subplots(2,4,figsize=(4.9,2.75),dpi=240)
for r,(Xd,yd,Xt_,yt_,lab) in enumerate(((XL,yL,XLt,yLt,'LDA'),(XQ,yQ,XQt,yQt,'QDA'))):
    for c_,a_ in enumerate(SH):
        ax=axs[r,c_]
        mdl=LDA(solver='lsqr',shrinkage=a_) if r==0 else QDA(reg_param=a_)
        mdl.fit(Xd,yd); acc=mdl.score(Xt_,yt_); NUM['gallery'][lab.lower()][str(a_)]=round(float(acc),4)
        ax.set_xlim(-3.6,5.4); ax.set_ylim(-3.6,4.2); gx,gy,G=grid(ax,n=200)
        ax.contourf(gx,gy,mdl.predict(G).reshape(gx.shape),levels=[-0.5,0.5,1.5],colors=[LC[0],LC[1]],alpha=0.55)
        ax.contour(gx,gy,mdl.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=1.6)
        dots(ax,Xd,yd,s=9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(('$a$' if r==0 else '$r$')+f' = {a_:g}',fontsize=9,color=PUR,pad=2)
        for sp_ in ax.spines.values(): sp_.set_visible(True)
axs[0,0].set_ylabel('LDA',fontsize=9.5,color=NAVY,weight='bold'); axs[1,0].set_ylabel('QDA',fontsize=9.5,color=NAVY,weight='bold')
fig.tight_layout(w_pad=0.3,h_pad=0.5); sv(fig,'f186.png',450)
# ---------------- f187 validation curves
warnings.simplefilter('ignore')
P_,N_,REP=50,40,20; rg=np.random.default_rng(0); idx=np.arange(P_); SgmA=0.6**np.abs(idx[:,None]-idx[None,:]); LA=np.linalg.cholesky(SgmA); dl=0.3*(-1.0)**idx
def draw(n):
    yy=np.arange(n)%2; return rg.normal(size=(n,P_))@LA.T+yy[:,None]*dl,yy
grid_a=[0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; accL=np.zeros((REP,len(grid_a))); acc_auto=[]; acc_svd=[]
for rep in range(REP):
    Xa_,ya_=draw(N_); Xb_,yb_=draw(4000)
    for j,a_ in enumerate(grid_a): accL[rep,j]=LDA(solver='lsqr',shrinkage=a_).fit(Xa_,ya_).score(Xb_,yb_)
    acc_auto.append(LDA(solver='lsqr',shrinkage='auto').fit(Xa_,ya_).score(Xb_,yb_)); acc_svd.append(LDA().fit(Xa_,ya_).score(Xb_,yb_))
mL=accL.mean(0); NUM['valcurve_lda']={'grid':grid_a,'mean':np.round(mL,4).tolist(),'auto':round(float(np.mean(acc_auto)),4),'svd':round(float(np.mean(acc_svd)),4),'best_a':grid_a[int(mL.argmax())],'best_acc':round(float(mL.max()),4)}
# QDA
P2,NK,REP2=10,25,30; rg2=np.random.default_rng(5); i2=np.arange(P2); S0q=0.5**np.abs(i2[:,None]-i2[None,:]); S1q=2.0*(-0.5)**np.abs(i2[:,None]-i2[None,:]); L0=np.linalg.cholesky(S0q); L1=np.linalg.cholesky(S1q); d2=0.6*np.ones(P2)
def draw2(nk): return np.r_[rg2.normal(size=(nk,P2))@L0.T,rg2.normal(size=(nk,P2))@L1.T+d2],np.r_[np.zeros(nk),np.ones(nk)].astype(int)
grid_r=[0.0,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]; accQ=np.zeros((REP2,len(grid_r))); accL2=[]
for rep in range(REP2):
    Xa_,ya_=draw2(NK); Xb_,yb_=draw2(3000)
    for j,r_ in enumerate(grid_r): accQ[rep,j]=QDA(reg_param=r_).fit(Xa_,ya_).score(Xb_,yb_)
    accL2.append(LDA().fit(Xa_,ya_).score(Xb_,yb_))
mQ=accQ.mean(0); NUM['valcurve_qda']={'grid':grid_r,'mean':np.round(mQ,4).tolist(),'lda':round(float(np.mean(accL2)),4),'best_r':grid_r[int(mQ.argmax())],'best_acc':round(float(mQ.max()),4)}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.45),dpi=240)
ax=axs[0]; sd=accL.std(0)
ax.fill_between(grid_a,mL-sd,mL+sd,color='#E6D9F0',alpha=0.8,lw=0); ax.plot(grid_a,mL,color=PUR,lw=1.8,marker='o',ms=3)
ax.axhline(np.mean(acc_auto),color=GREEN,lw=1.3,ls='--'); ax.text(0.2,0.62,'dashed: shrinkage="auto"',color=GREEN,fontsize=8.5,ha='left',va='center')
ax.axhline(np.mean(acc_svd),color=GREY,lw=1.1,ls=':'); ax.text(0.2,0.55,'dotted: svd solver',color=GREY,fontsize=8.5,ha='left',va='center')
ax.set_xlabel('shrinkage $a$ (lsqr solver)'); ax.set_ylabel('test accuracy'); ax.set_title('LDA, $p$ = 50, $n$ = 40',fontsize=9,color=NAVY); ax.set_ylim(0.42,0.93); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax=axs[1]; sdq=accQ.std(0); ax.fill_between(grid_r,mQ-sdq,mQ+sdq,color='#E6D9F0',alpha=0.8,lw=0); ax.plot(grid_r,mQ,color=PUR,lw=1.8,marker='o',ms=3)
ax.axhline(np.mean(accL2),color=GREY,lw=1.1,ls=':'); ax.text(0.02,np.mean(accL2)+0.006,'LDA (shared covariance)',color=GREY,fontsize=8.5,ha='left',va='bottom')
ax.set_xlabel('QDA reg_param $r$'); ax.set_title('QDA, $p$ = 10, 25 per class',fontsize=9,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f187.png',450)
# ---------------- f188 case study
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
dc=load_breast_cancer(); Xt_,Xe_,yt_,ye_=train_test_split(dc.data,dc.target,test_size=0.25,random_state=0,stratify=dc.target)
pl=make_pipeline(StandardScaler(),LDA()).fit(Xt_,yt_); pr=make_pipeline(StandardScaler(),LogisticRegression()).fit(Xt_,yt_)
sl=pl.decision_function(Xe_); sr=pr.decision_function(Xe_)
NUM['case']={'lda_score_min':round(float(sl.min()),2),'lda_score_max':round(float(sl.max()),2),'lr_score_min':round(float(sr.min()),2),'lr_score_max':round(float(sr.max()),2),
  'wrong_lda':int(((sl>0)!=(ye_==1)).sum()),'wrong_lr':int(((sr>0)!=(ye_==1)).sum()),'both_wrong':int((((sl>0)!=(ye_==1))&((sr>0)!=(ye_==1))).sum()),'corr_scores':round(float(np.corrcoef(sl,sr)[0,1]),3),
  'abs_gt10_lda':int((np.abs(sl)>10).sum()),'abs_gt10_lr':int((np.abs(sr)>10).sum())}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.45),dpi=240)
ax=axs[0]; bins=np.linspace(-22,22,23)
for k,nm in ((0,'malignant'),(1,'benign')): ax.hist(sl[ye_==k],bins=bins,color=LC[k],ec=C3[k],lw=0.7,alpha=0.9,histtype='stepfilled',label=nm)
ax.axvline(0,color=GREEN,lw=1.6); ax.set_xlabel('LDA score $\\delta_1-\\delta_0$ (log-odds of benign)'); ax.set_ylabel('test tumours'); ax.legend(fontsize=8.5,loc='upper left',handlelength=1.2); ax.set_title('LDA scores, test set',fontsize=9,color=NAVY)
ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax=axs[1]
for k in (0,1): ax.scatter(sr[ye_==k],sl[ye_==k],s=11,color=C3[k],marker=M3[k],edgecolor='white',lw=0.4,zorder=4)
ax.plot([-30,30],[-30,30],color=GRID,lw=1.0,zorder=1); ax.axhline(0,color=GREEN,lw=0.8); ax.axvline(0,color=GREEN,lw=0.8)
ax.set_xlim(-30,17); ax.set_ylim(-23,23); ax.set_xlabel('logistic score (log-odds)'); ax.set_ylabel('LDA score'); ax.set_title('Same tumours, two models',fontsize=9,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f188.png',450)
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1); print(json.dumps(NUM)); print(json.dumps(CHK))

# All figures of Chapter 30. Run: python3 fig/figs.py  (PNGs into ch30/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse
from scipy.special import expit
from sklearn.datasets import make_moons
from sklearn.gaussian_process import GaussianProcessRegressor as GPR, GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel as C
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore',category=RuntimeWarning)
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
from matplotlib.colors import LinearSegmentedColormap
BO=LinearSegmentedColormap.from_list('bo',[BLUE,'#DCE7F7','#FFFFFF','#FCE5C2',OR])
NUM={}; CHK={}
LG='#CFE8D8'
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grd(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx,yy,s=30,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.7,zorder=6,**k)
# ---------------------------------------------------------------- shared toy data (Listing 30.4)
TX=np.array([1.,2,3,5,6,8]); TY=np.array([0.2,1.1,1.5,0.4,-0.6,0.3])
gp=GPR(C(1.0)*RBF(1.0),alpha=0.01,n_restarts_optimizer=5,random_state=0).fit(TX[:,None],TY)
th=np.exp(gp.kernel_.theta); SF2,ELL=float(th[0]),float(th[1]); NUM['toy_kernel']=[round(SF2,4),round(ELL,4)]
KFIX=lambda: C(SF2,'fixed')*RBF(ELL,'fixed')
grid=np.linspace(0,10,300)
def band(ax,m,s,k=2,color=LG): ax.fill_between(grid,m-k*s,m+k*s,color=color,lw=0,zorder=1)

# ---- 30.0-1 opener
fig,ax=plt.subplots(figsize=(5.0,2.2),dpi=240)
m,s=gp.predict(grid[:,None],return_std=True); band(ax,m,s)
S=gp.sample_y(grid[:,None],n_samples=5,random_state=2)
for j in range(5): ax.plot(grid,S[:,j],color='#6DB38A',lw=0.8,ls=(0,(4,1.5)),zorder=2)
ax.plot(grid,m,color=GREEN,lw=2.2,zorder=4); pts(ax,TX,TY,s=40)
ax.set_xlim(0,10); ax.set_ylim(-1.9,2.5); grd(ax); ax.set_xlabel('setting $x$'); ax.set_ylabel('response $y$')
ax.text(9.95,2.3,'mean prediction',color=GREEN,ha='right',va='top',fontsize=9.5,weight='bold')
ax.annotate('band: mean $\\pm$ 2 standard deviations',xy=(4.0,m[120]+2*s[120]-0.0),xytext=(0.3,2.3),fontsize=8.8,color=GREEN,va='top',arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.7,relpos=(0.88,0.0),shrinkA=1,shrinkB=0))
ax.text(0.15,-1.75,'dashed: functions the data still allow',color='#4F9A6E',fontsize=8.8,va='bottom')
ax.text(1.15,-0.1,'six measurements',color=BLUE,fontsize=9,ha='left')
sv(fig,'f3000.png',470)

# ---- 30.1-1 the six readings and two questions
fig,ax=plt.subplots(figsize=(3.6,2.1),dpi=240); pts(ax,TX,TY,s=44)
for a,b_ in zip(TX,TY): ax.text(a,b_+(0.22 if b_>-0.5 else -0.42),f'{b_:g}'.replace('-','\u2212'),fontsize=8.5,color=GREY,ha='center')
for xq,t in [(4,'$x = 4$: ?'),(9.5,'$x = 9.5$: ?')]:
    ax.axvline(xq,color=PUR,lw=1,ls='--'); ax.text(xq+(0.1 if xq<9 else -0.1),-1.45,t,color=PUR,fontsize=8.8,ha='left' if xq<9 else 'right')
ax.set_xlim(0,10.5); ax.set_ylim(-1.7,2.1); grd(ax); ax.set_xlabel('setting $x$'); ax.set_ylabel('response $y$'); sv(fig,'f3011.png',340)

# ---- 30.2-1 prior samples and the pair of values
g2=np.linspace(0,8,160); Kp=RBF(1.0)(g2[:,None]); rng=np.random.default_rng(5)
Sp=rng.multivariate_normal(np.zeros(160),Kp+1e-8*np.eye(160),5)
fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240,gridspec_kw={'width_ratios':[1.3,1]})
ax=axs[0]
for j,c in enumerate([BLUE,OR,GREEN,PUR,RED]): ax.plot(g2,Sp[j],color=c,lw=1.1,ls=['-','--','-.',':','-'][j])
for xv in (3.0,3.6): ax.axvline(xv,color=GREY,lw=0.7,ls=':')
ax.text(3.3,2.75,'$x_1, x_2$',ha='center',fontsize=9,color=GREY); ax.set_ylim(-2.9,3.1); ax.set_xlim(0,8); grd(ax); ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$')
ax=axs[1]; Z=rng.multivariate_normal(np.zeros(3),np.exp(-0.5*np.subtract.outer([3.0,3.6,6.0],[3.0,3.6,6.0])**2),400)
ax.scatter(Z[:,0],Z[:,1],s=6,color=BLUE,alpha=0.6,lw=0,label='$x_2 = 3.6$'); ax.scatter(Z[:,0],Z[:,2],s=6,color=OR,alpha=0.6,lw=0,marker='^',label='$x_2 = 6.0$')
ax.set_xlim(-3.2,3.2); ax.set_ylim(-3.2,3.6); grd(ax); ax.set_xlabel('$f(x_1)$, $x_1 = 3.0$'); ax.set_ylabel('$f(x_2)$'); ax.legend(loc='upper left',fontsize=8,markerscale=2.5,handletextpad=0.1,borderaxespad=0.1)
rho1=float(np.exp(-0.5*0.6**2)); rho2=float(np.exp(-0.5*3**2)); NUM['rho']=[round(rho1,4),round(rho2,4)]
NUM['sample_corr']=[round(float(np.corrcoef(Z[:,0],Z[:,1])[0,1]),3),round(float(np.corrcoef(Z[:,0],Z[:,2])[0,1]),3)]
fig.tight_layout(w_pad=0.8); sv(fig,'f3021.png',440)

# ---- 30.2-2 learning by conditioning
fig,axs=plt.subplots(1,3,figsize=(5.0,1.95),dpi=240,sharey=True)
for ax,idx,tt in zip(axs,[[],[1,3],[0,1,2,3,4,5]],['prior: no data','two readings','six readings']):
    g=GPR(KFIX(),alpha=0.01,optimizer=None)
    if idx: g.fit(TX[idx][:,None],TY[idx])
    m_,s_=g.predict(grid[:,None],return_std=True); band(ax,m_,s_)
    Sm=g.sample_y(grid[:,None],n_samples=3,random_state=4)
    for j,c in enumerate(['#6DB38A']*3): ax.plot(grid,Sm[:,j],color=c,lw=0.8,ls=(0,(4,1.5)))
    ax.plot(grid,m_,color=GREEN,lw=1.8,zorder=4)
    if idx: pts(ax,TX[idx],TY[idx],s=26)
    ax.set_title(tt,fontsize=9.5,color=NAVY); ax.set_xlim(0,10); ax.set_xticks([0,5,10]); ax.set_ylim(-2.7,2.7); grd(ax); ax.set_xlabel('$x$')
axs[0].set_ylabel('$f(x)$'); fig.tight_layout(w_pad=0.5); sv(fig,'f3022.png',470)

# ---- 30.3-1 conditioning geometry
rho=0.8; fig,ax=plt.subplots(figsize=(3.7,2.7),dpi=240)
for k,ls_ in [(1,'-'),(2,'--')]:
    ev,evec=np.linalg.eigh(np.array([[1,rho],[rho,1]])); ang=np.degrees(np.arctan2(evec[1,1],evec[0,1]))
    ax.add_patch(Ellipse((0,0),2*k*np.sqrt(ev[1]),2*k*np.sqrt(ev[0]),angle=ang,fill=False,ec=BLUE,lw=1.1,ls=ls_))
a0=1.5; bb=np.linspace(-3,3,200); cm,cs=rho*a0,np.sqrt(1-rho**2)
pdfc=np.exp(-0.5*((bb-cm)/cs)**2)/(cs*np.sqrt(2*np.pi)); pdfm=np.exp(-0.5*bb**2)/np.sqrt(2*np.pi)
ax.axvline(a0,color=RED,lw=1.1); ax.plot(a0+0.9*pdfc,bb,color=GREEN,lw=1.8); ax.plot(-2.95+0.9*pdfm,bb,color=GREY,lw=1.3,ls='--')
ax.plot([a0],[cm],'o',color=GREEN,ms=5.5,mec='white',zorder=6); ax.plot([a0],[0],'o',color=RED,ms=0)
ax.text(a0+0.08,-2.75,'observe\n$f_1 = 1.5$',color=RED,fontsize=8.8,va='bottom',ha='left'); ax.text(-1.95,-2.75,'prior of $f_2$',color=GREY,fontsize=8.8,ha='left',va='bottom')
ax.annotate('conditional of $f_2$:\nmean 1.2,\nstandard deviation 0.6',xy=(a0+0.55,cm+0.45),xytext=(2.15,2.35),color=GREEN,fontsize=8.8,ha='left',va='center',arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.7))
ax.set_xlim(-3.0,4.4); ax.set_ylim(-3,3); ax.set_xlabel('$f_1$'); ax.set_ylabel('$f_2$'); grd(ax); ax.set_aspect('equal'); sv(fig,'f3031.png',350)
NUM['cond']=[rho*a0,float(cs)]

# ---- 30.5-1 gallery of kernels
kers=[('RBF, $\\ell=1$',RBF(1.0)),('Mat\u00e9rn $\\nu=\\frac{1}{2}$',Matern(1.0,nu=0.5)),('Mat\u00e9rn $\\nu=\\frac{3}{2}$',Matern(1.0,nu=1.5)),('Mat\u00e9rn $\\nu=\\frac{5}{2}$',Matern(1.0,nu=2.5)),
 ('RationalQuadratic',RationalQuadratic(1.0,0.5)),('ExpSineSquared',ExpSineSquared(1.0,3.0)),('DotProduct',DotProduct(1.0)),('ExpSineSquared\n$\\times$ RBF',ExpSineSquared(1.0,3.0)*RBF(4.0))]
g8=np.linspace(0,10,160)
fig,axs=plt.subplots(2,4,figsize=(5.0,3.0),dpi=240,sharex=True)
for ax,(tt,k_),c in zip(axs.ravel(),kers,[BLUE,BLUE,BLUE,BLUE,PUR,PUR,PUR,PUR]):
    Sk=GPR(k_).sample_y(g8[:,None],n_samples=3,random_state=(8 if 'Dot' in tt else 3))
    for j,ls_ in enumerate(['-','--','-.']): ax.plot(g8,Sk[:,j],color=[BLUE,OR,GREEN][j],lw=1.0,ls=ls_)
    ax.set_title(tt,fontsize=8.6,color=NAVY,pad=3); ax.set_xlim(0,10); ax.set_xticks([0,5,10]); ax.grid(color=GRID,lw=0.4); ax.set_axisbelow(True); ax.tick_params(labelsize=8.2)
    ax.set_ylim(-3.2,3.2) if 'Dot' not in tt else ax.set_ylim(-11,11)
for ax in axs[1]: ax.set_xlabel('$x$',fontsize=9)
fig.tight_layout(h_pad=0.5,w_pad=0.3); sv(fig,'f3051.png',470)

# ---- 30.7-1 flow chart
fig,ax=new(6.4,5.6)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.6): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.5,7.4,5.6,0.55,'Input: $\\mathbf{X}$ ($n$ rows), $\\mathbf{y}$;  kernel, $\\alpha$, normalize_y, optimizer',fc=LGREY,ec=GREY)
fb(3.5,6.5,5.6,0.55,'If normalize_y: $\\mathbf{y}\\leftarrow(\\mathbf{y}-\\bar{y})/s_y$')
ax.add_patch(Polygon([(3.5,6.0),(5.2,5.5),(3.5,5.0),(1.8,5.5)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.5,5.5,'optimizer\nnot None?',ha='center',va='center',fontsize=8.4,color=NAVY)
fb(5.4,4.05,2.8,1.4,'yes: maximise the log\nmarginal likelihood over\n$\\theta$ (L-BFGS-B, bounds;\nextra random restarts)',fc='#F1EAF7',ec=PUR,fs=8.7)
fb(1.55,4.05,2.7,1.4,'no: keep the\nhyperparameters\nof the kernel as given',fc='#F1EAF7',ec=PUR,fs=8.7)
fb(3.5,2.75,5.6,0.55,'Build $\\mathbf{K}=k(\\mathbf{X},\\mathbf{X})$; add $\\alpha$ to the diagonal  (Eq. 30.3-3)')
fb(3.5,1.8,5.6,0.65,'Cholesky factor  $\\mathbf{K}+\\alpha\\mathbf{I}=\\mathbf{L}\\mathbf{L}^{\\mathrm{T}}$  ($n^3/3$ operations);\n$\\mathbf{a}=\\mathbf{L}^{\\mathrm{T}}\\backslash(\\mathbf{L}\\backslash\\mathbf{y})$',fs=8.7)
fb(3.5,0.85,5.6,0.55,'Store  kernel_,  L_,  alpha_,  log_marginal_likelihood_value_',fc=LGREY,ec=GREY,fs=7.9)
fb(3.5,-0.2,5.6,0.95,'predict:  $\\mathbf{k}_*=k(\\mathbf{X}_*,\\mathbf{X})$;  mean $=\\mathbf{k}_*^{\\mathrm{T}}\\mathbf{a}$  ($n$ per point)\n$\\mathbf{v}=\\mathbf{L}\\backslash\\mathbf{k}_*^{\\mathrm{T}}$;  std $=\\sqrt{k(x_*,x_*)-\\mathbf{v}^{\\mathrm{T}}\\mathbf{v}}$  ($n^2$ per point)',fc='#E3F4EA',ec=GREEN,fs=8.7)
A((3.5,7.12),(3.5,6.78)); A((3.5,6.22),(3.5,6.0))
ax.plot([5.2,5.4],[5.5,5.5],color=NAVY,lw=0.9); A((5.4,5.5),(5.4,4.72)); ax.plot([1.8,1.55],[5.5,5.5],color=NAVY,lw=0.9); A((1.55,5.5),(1.55,4.72))
ax.plot([5.4,5.4],[3.4,3.2],color=NAVY,lw=0.9); ax.plot([1.55,1.55],[3.4,3.2],color=NAVY,lw=0.9)
ax.plot([5.4,3.5],[3.2,3.2],color=NAVY,lw=0.9); ax.plot([1.55,3.5],[3.2,3.2],color=NAVY,lw=0.9); A((3.5,3.2),(3.5,3.03))
A((3.5,2.47),(3.5,2.15)); A((3.5,1.47),(3.5,1.15)); A((3.5,0.57),(3.5,0.3))
ax.set_xlim(0,7); ax.set_ylim(-0.8,7.8); sv(fig,'f3071.png',380)

# ---- dense data of Section 30.11
rng=np.random.default_rng(1); dx=np.sort(rng.uniform(0,10,14)); dy=np.sin(0.9*dx)+0.35*rng.normal(size=14); DX=dx[:,None]
def lml(l,n_,sf=1.0):
    g=GPR(C(sf**2,'fixed')*RBF(l,'fixed'),alpha=n_**2,optimizer=None).fit(DX,dy); return g.log_marginal_likelihood_value_
fig,axs=plt.subplots(2,3,figsize=(5.0,3.2),dpi=240,sharey=True,sharex=True); NUM['gal']={}
cfg=[(0.3,0.31),(1.4,0.31),(6.0,0.31),(1.4,0.05),(1.4,0.31),(1.4,1.0)]
for ax,(l,n_) in zip(axs.ravel(),cfg):
    g=GPR(C(1.0,'fixed')*RBF(l,'fixed'),alpha=n_**2,optimizer=None).fit(DX,dy); m_,s_=g.predict(grid[:,None],return_std=True); band(ax,m_,s_)
    ax.plot(grid,m_,color=GREEN,lw=1.6,zorder=4); pts(ax,dx,dy,s=14)
    ax.set_xlim(0,10); ax.set_ylim(-2.6,2.6); grd(ax); ax.tick_params(labelsize=8.5)
    ax.set_title('$\\ell=%g$,  $\\sigma_n=%g$' % (l,n_),fontsize=9,color=PUR,pad=3); ax.set_xticks([0,5,10])
    ax.text(9.8,-2.45,'LML $=%.1f$' % g.log_marginal_likelihood_value_,fontsize=8.8,ha='right',color=NAVY); NUM['gal'][f'{l}_{n_}']=round(float(g.log_marginal_likelihood_value_),2)
for ax in axs[1]: ax.set_xlabel('$x$',fontsize=9)
for ax in axs[:,0]: ax.set_ylabel('$y$',fontsize=9)
fig.tight_layout(h_pad=0.6,w_pad=0.3); sv(fig,'f3091.png',470)

# ---- 30.11-2 likelihood landscape and the three terms
ls_=np.exp(np.linspace(np.log(0.12),np.log(8),70)); ns=np.exp(np.linspace(np.log(0.03),np.log(1.5),70)); Zg=np.array([[lml(l,n_) for l in ls_] for n_ in ns])
opt=[]
for l0,n0 in [(0.3,0.05**2),(1.4,0.3**2)]:
    gk=GPR(C(1.0,'fixed')*RBF(l0)+WhiteKernel(n0),alpha=1e-10,random_state=0).fit(DX,dy); pr=gk.kernel_.get_params()
    opt.append((float([v for k,v in pr.items() if k.endswith('length_scale')][0]),float(np.sqrt([v for k,v in pr.items() if k.endswith('noise_level')][0])),float(gk.log_marginal_likelihood_value_)))
NUM['optima']=[[round(v,3) for v in o] for o in opt]
fig,axs=plt.subplots(1,2,figsize=(5.0,2.1),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; cs=ax.contourf(np.log10(ls_),np.log10(ns),Zg,levels=[-40,-30,-24,-20,-17,-15,-13,-12,-11.4,-11.0],cmap='Blues',extend='both'); ax.contour(np.log10(ls_),np.log10(ns),Zg,levels=[-24,-17,-13,-12],colors='white',linewidths=0.5)
for (l,n_,v),c in zip(opt,[RED,GREEN]): ax.plot(np.log10(l),np.log10(n_),'*',color=c,ms=11,mec='white',mew=0.7,zorder=5)
ax.text(np.log10(opt[0][0])+0.1,np.log10(opt[0][1])+0.0,'peak 1',color=RED,fontsize=8.5,weight='bold',ha='left'); ax.text(np.log10(opt[1][0])+0.12,np.log10(opt[1][1])+0.12,'peak 2',color=GREEN,fontsize=8.5,weight='bold',ha='left')
ax.set_xticks(np.log10([0.2,0.5,1,2,5])); ax.set_xticklabels(['0.2','0.5','1','2','5']); ax.set_yticks(np.log10([0.05,0.1,0.3,1])); ax.set_yticklabels(['0.05','0.1','0.3','1'])
ax.set_xlabel('length-scale $\\ell$'); ax.set_ylabel('noise std. dev. $\\sigma_n$'); ax.set_title('log marginal likelihood',fontsize=9.5,color=NAVY)
ax=axs[1]; n_=opt[1][1]; lv=np.exp(np.linspace(np.log(0.15),np.log(8),90)); T=[]
for l in lv:
    g=GPR(C(1.0,'fixed')*RBF(l,'fixed'),alpha=n_**2,optimizer=None).fit(DX,dy); Ls=g.L_; al=g.alpha_
    T.append((-0.5*dy@al,-np.log(np.diag(Ls)).sum(),-0.5*14*np.log(2*np.pi)))
T=np.array(T); ax.plot(lv,T[:,0],color=BLUE,lw=1.5,label='data fit',ls='-'); ax.plot(lv,T[:,1],color=RED,lw=1.5,label='complexity',ls='--'); ax.plot(lv,T[:,2],color=GREY,lw=1.2,label='constant',ls=':'); ax.plot(lv,T.sum(1),color=GREEN,lw=2.2,label='sum')
ax.set_xscale('log'); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xticks([0.2,0.5,1,2,5]); ax.set_xticklabels(['0.2','0.5','1','2','5']); ax.set_xlabel('length-scale $\\ell$  ($\\sigma_n=0.31$)'); ax.set_ylim(-34,16)
ax.legend(loc='lower left',fontsize=8.6,labelspacing=0.2,handlelength=1.6,ncol=2,columnspacing=0.8,bbox_to_anchor=(0.0,0.0)); grd(ax); ax.set_title('the three terms, Eq. 30.4-2',fontsize=9.5,color=NAVY)
NUM['terms_at']={str(round(l,2)):[round(float(a),2) for a in T[i]] for i,l in enumerate(lv) if i in (0,30,50,89)}
fig.tight_layout(w_pad=0.8); sv(fig,'f3092.png',470)

# ---- 30.10-1 GPC in one dimension
Xc=np.array([[-3.],[-2],[-1],[0.5],[1],[2],[3]]); yc=np.array([0,0,0,1,0,1,1]); kc=C(4.0,'fixed')*RBF(1.5,'fixed'); gc=GPC(kc).fit(Xc,yc)
gx=np.linspace(-6,6,300)[:,None]; lm,lv_=gc.base_estimator_.latent_mean_and_variance(gx); pp=gc.predict_proba(gx)[:,1]; fh=np.log(gc.base_estimator_.pi_/(1-gc.base_estimator_.pi_))
fig,axs=plt.subplots(1,2,figsize=(4.7,2.1),dpi=240)
ax=axs[0]; ax.fill_between(gx[:,0],lm-2*np.sqrt(lv_),lm+2*np.sqrt(lv_),color=LG,lw=0); ax.plot(gx[:,0],lm,color=GREEN,lw=1.9); ax.axhline(0,color=GREY,lw=0.7)
ax.scatter(Xc[yc==0,0],fh[yc==0],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=6); ax.scatter(Xc[yc==1,0],fh[yc==1],s=34,color=OR,marker='s',edgecolor='white',lw=0.7,zorder=6)
ax.set_xlabel('$x$'); ax.set_ylabel('latent function $f$'); ax.set_xticks([-5,0,5]); ax.set_xlim(-6,6); ax.set_ylim(-3.3,3.3); grd(ax); ax.set_title('latent mean $\\pm$ 2 std. dev.',fontsize=9.5,color=NAVY)
ax=axs[1]; ax.plot(gx[:,0],expit(lm),color=GREY,lw=1.2,ls='--',label='$\\sigma$(latent mean)'); ax.plot(gx[:,0],pp,color=GREEN,lw=1.9,label='$P(y=1\\,|\\,x)$')
ax.scatter(Xc[yc==0,0],yc[yc==0],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=6); ax.scatter(Xc[yc==1,0],yc[yc==1],s=34,color=OR,marker='s',edgecolor='white',lw=0.7,zorder=6)
ax.axhline(0.5,color=GREY,lw=0.5); ax.set_xlabel('$x$'); ax.set_ylabel('probability'); ax.set_xticks([-5,0,5]); ax.set_xlim(-6,6); ax.set_ylim(-0.04,1.04); grd(ax); ax.legend(loc='center left',fontsize=8.6,bbox_to_anchor=(0.0,0.72),handlelength=1.6); ax.set_title('class probability',fontsize=9.5,color=NAVY)
NUM['gpc1d']={'p_at_6':round(float(pp[-1]),3),'sigma_mean_at_6':round(float(expit(lm[-1])),3),'p_at_-6':round(float(pp[0]),3)}
fig.tight_layout(w_pad=0.8); sv(fig,'f3101.png',440)

# ---- 30.13 case 1: the seasonal series
cs_=json.load(open('verify/case_seasonal.json')); t=np.array(cs_['t']); yy=np.array(cs_['y'])
fig,axs=plt.subplots(1,2,figsize=(5.0,2.2),dpi=240,sharey=True)
for ax,name,tt in zip(axs,['trend + season + noise','RBF + noise'],['trend + season + noise','RBF + noise']):
    m_=np.array(cs_[name]['mean']); s_=np.array(cs_[name]['std'])
    ax.fill_between(t,m_-2*s_,m_+2*s_,color=LG,lw=0,zorder=1); ax.plot(t,m_,color=GREEN,lw=1.5,zorder=4)
    ax.scatter(t[:72],yy[:72],s=6,color=BLUE,lw=0,zorder=5); ax.scatter(t[72:],yy[72:],s=9,color=GREY,marker='D',lw=0,zorder=5)
    ax.axvline(6-1/24,color=PUR,lw=0.9,ls='--'); ax.set_title(tt,fontsize=9,color=NAVY); ax.set_xlim(0,8); ax.set_xticks([0,2,4,6,8]); ax.set_ylim(-3.5,8.2); grd(ax); ax.set_xlabel('time (years)')
axs[0].set_ylabel('$y$'); axs[0].text(5.9,-3.2,'training | test',fontsize=8.6,color=PUR,ha='right',va='bottom'); axs[1].text(5.9,-3.2,'training | test',fontsize=8.6,color=PUR,ha='right',va='bottom')
fig.tight_layout(w_pad=0.6); sv(fig,'f3131.png',470)

# ---- 30.13 case 2: two moons
X,y=make_moons(n_samples=200,noise=0.25,random_state=0); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.5,random_state=0,stratify=y)
g_=GPC(C(1.0)*RBF(1.0),random_state=0).fit(Xtr,ytr); lr=LogisticRegression().fit(Xtr,ytr)
gx0,gy0=np.meshgrid(np.linspace(-3,4,130),np.linspace(-2.5,3.5,130)); G2=np.c_[gx0.ravel(),gy0.ravel()]
fig,axs=plt.subplots(1,2,figsize=(5.0,2.4),dpi=240,sharey=True)
for ax,mdl,tt in zip(axs,[g_,lr],['Gaussian process classifier','logistic regression']):
    Pm=mdl.predict_proba(G2)[:,1].reshape(gx0.shape); cf=ax.contourf(gx0,gy0,Pm,levels=np.linspace(0,1,11),cmap=BO,alpha=0.9,vmin=0,vmax=1); ax.contour(gx0,gy0,Pm,levels=[0.5],colors=GREEN,linewidths=1.8)
    ax.scatter(Xtr[ytr==0,0],Xtr[ytr==0,1],s=11,color=BLUE,edgecolor='white',lw=0.5,zorder=5); ax.scatter(Xtr[ytr==1,0],Xtr[ytr==1,1],s=13,color=OR,marker='s',edgecolor='white',lw=0.5,zorder=5)
    ax.set_title(tt,fontsize=9,color=NAVY); ax.set_xlabel('$x_1$'); ax.set_xticks([-2,0,2,4]); ax.set_xlim(-3,4); ax.set_ylim(-2.5,3.5)
axs[0].set_ylabel('$x_2$'); fig.tight_layout(w_pad=0.5); fig.subplots_adjust(right=0.92); cax=fig.add_axes([0.935,0.2,0.016,0.6]); cb=fig.colorbar(cf,cax=cax); cb.set_label('$P(y=1)$',fontsize=8.5); cb.ax.tick_params(labelsize=7.5)
sv(fig,'f3132.png',470)
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_fontcheck.json','w'),indent=1)
print(json.dumps(NUM)); print({k:(v['min_font_printed']) for k,v in CHK.items()})

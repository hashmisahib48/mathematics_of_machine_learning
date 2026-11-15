# Problem figures of Chapter 30. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..'); sys.path.insert(0,'code')
from figstyle import *
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from gp_scratch import MyGP
import matplotlib.text as mtext
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9,'legend.fontsize':9})
A={}; LG='#CFE8D8'; CHK={}
def small(w=2.4,h=1.05): return plt.subplots(figsize=(w,h),dpi=240)
def grd(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name); CHK[name]=fs
TX=np.array([1.,2,3,5,6,8]); TY=np.array([0.2,1.1,1.5,0.4,-0.6,0.3]); SF2,ELL=0.7366,1.3672
k=lambda a,b,l=1.0: np.exp(-(a[:,None]-b[None,:])**2/(2*l**2))
# ---- 30A.1 a covariance matrix
x=np.array([0.,1,3]); K1=k(x,x); K2=k(x,x,2.0)
A['30A.1']={'K_ell1':np.round(K1,4).tolist(),'k03_ell2':round(float(K2[0,2]),4),'corr_ell1':round(float(K1[0,2]),4)}
fig,ax=small(2.4,1.2); ax.imshow(K1,cmap='Blues',vmin=0,vmax=1.2,aspect='auto')
for i in range(3):
    for j in range(3): ax.text(j,i,f'{K1[i,j]:.3f}',ha='center',va='center',fontsize=9,color='white' if K1[i,j]>0.6 else NAVY)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['0','1','3']); ax.set_yticklabels(['0','1','3']); ax.set_xlabel('$x_j$'); ax.set_ylabel('$x_i$'); ax.xaxis.set_label_position('bottom')
for s in ax.spines.values(): s.set_visible(False)
sv(fig,'p_30A1.png')
# ---- 30A.2 conditioning two variables
rho=0.8; s2=0.25; y1=1.5; m0,v0=rho*y1,1-rho**2; m1=rho/(1+s2)*y1; v1=1-rho**2/(1+s2)
A['30A.2']={'noise_free':[m0,round(v0,4)],'noisy':[round(m1,4),round(v1,4)]}
fig,ax=small(2.4,1.0); tt=np.linspace(-3,3.5,300); pdf=lambda m,v: np.exp(-0.5*(tt-m)**2/v)/np.sqrt(2*np.pi*v)
ax.plot(tt,pdf(0,1),color=GREY,lw=1.2,ls='--',label='prior'); ax.plot(tt,pdf(m0,v0),color=GREEN,lw=1.8,label='exact $y_1$'); ax.plot(tt,pdf(m1,v1),color=OR,lw=1.8,ls='-.',label='noisy $y_1$')
ax.set_xlabel('$f_2$'); ax.set_yticks([]); ax.legend(fontsize=9,loc='upper left',handlelength=1.6,borderaxespad=0.1,labelspacing=0.2); ax.set_xlim(-3,3.5); ax.set_ylim(0,1.15); sv(fig,'p_30A2.png')
# ---- 30A.3 one reading, one question
sn=0.25; y0=1.5
def one(xs): ks=np.exp(-xs**2/2); return ks*y0/(1+sn), 1-ks**2/(1+sn)
A['30A.3']={'x0':[round(v,4) for v in one(0.0)],'x1':[round(float(v),4) for v in one(1.0)],'x3':[round(float(v),4) for v in one(3.0)]}
fig,ax=small(2.4,1.5); g=np.linspace(-3.5,4.5,200); mm,vv=one(g); sd=np.sqrt(vv)
ax.fill_between(g,mm-2*sd,mm+2*sd,color=LG,lw=0); ax.plot(g,mm,color=GREEN,lw=1.8); ax.scatter([0],[y0],s=34,color=BLUE,edgecolor='white',zorder=6); ax.set_xlabel('$x_*$'); ax.set_ylabel('$f(x_*)$'); grd(ax); ax.set_xlim(-3.5,4.5); ax.set_ylim(-2.4,3.2); sv(fig,'p_30A3.png')
# ---- 30B.1 posterior samples by Cholesky
gp=MyGP(SF2,ELL,0.01).fit(TX[:,None],TY); grid=np.linspace(0,10,120); mean,cov=gp.predict(grid[:,None],return_cov=True)
Lc=np.linalg.cholesky(cov+1e-8*np.eye(120)); Z=np.random.default_rng(0).normal(size=(120,2000)); Sm=mean[:,None]+Lc@Z
sd_a=np.sqrt(np.diag(cov))
A['30B.1']={'max_mean_err':round(float(np.abs(Sm.mean(1)-mean).max()),4),'max_std_err':round(float(np.abs(Sm.std(1)-sd_a).max()),4),'n_draws':2000}
fig,ax=small(2.4,1.3); band=(mean-2*sd_a,mean+2*sd_a); ax.fill_between(grid,*band,color=LG,lw=0)
for j in range(4): ax.plot(grid,Sm[:,j],lw=0.8,color=['#6DB38A','#4F9A6E','#7FBF9B','#3F8A5E'][j],ls=['-','--','-.',':'][j])
ax.plot(grid,mean,color=GREEN,lw=1.8); ax.scatter(TX,TY,s=18,color=BLUE,edgecolor='white',lw=0.6,zorder=6); ax.set_xlabel('$x$'); ax.set_ylabel('$f$'); ax.set_xlim(0,10); ax.set_xticks([0,5,10]); grd(ax); sv(fig,'p_30B1.png')
# ---- 30B.2 alpha versus WhiteKernel
ga=GPR(C(SF2,'fixed')*RBF(ELL,'fixed'),alpha=0.04,optimizer=None).fit(TX[:,None],TY)
gw=GPR(C(SF2,'fixed')*RBF(ELL,'fixed')+WhiteKernel(0.04,'fixed'),alpha=1e-10,optimizer=None).fit(TX[:,None],TY)
ma,sa=ga.predict(grid[:,None],return_std=True); mw,sw=gw.predict(grid[:,None],return_std=True)
i3=np.argmin(abs(grid-3.0)); i4=np.argmin(abs(grid-4.0))
A['30B.2']={'max_mean_diff':float(np.abs(ma-mw).max()),'std_alpha_x3':round(float(sa[i3]),4),'std_white_x3':round(float(sw[i3]),4),'std_alpha_x4':round(float(sa[i4]),4),'std_white_x4':round(float(sw[i4]),4),'grid_x':[round(float(grid[i3]),3),round(float(grid[i4]),3)],'lml_equal':bool(np.isclose(ga.log_marginal_likelihood_value_,gw.log_marginal_likelihood_value_))}
fig,ax=small(2.4,1.35); ax.fill_between(grid,mw-2*sw,mw+2*sw,color='#F8DDB0',lw=0); ax.fill_between(grid,ma-2*sa,ma+2*sa,color=LG,lw=0.0,alpha=0.9); ax.plot(grid,ma,color=GREEN,lw=1.6); ax.scatter(TX,TY,s=18,color=BLUE,edgecolor='white',lw=0.6,zorder=6)
ax.text(9.9,4.4,'alpha: green',color=GREEN,fontsize=9,ha='right'); ax.text(9.9,3.6,'WhiteKernel: orange',color='#C77D0A',fontsize=9,ha='right')
ax.set_xlabel('$x$'); ax.set_ylabel('$\\pm 2$ std. dev.'); ax.set_xlim(0,10); ax.set_ylim(-2.6,5.2); ax.set_xticks([0,5,10]); grd(ax); sv(fig,'p_30B2.png')
# ---- 30B.3 normalize_y
Y100=TY+100; far=np.array([[30.0]]); res={}
for nm,flag in [('plain',False),('normalized',True)]:
    g=GPR(C(1.0)*RBF(1.5),alpha=0.01,normalize_y=flag,random_state=0,n_restarts_optimizer=3).fit(TX[:,None],Y100); res[nm]=[float(g.predict(far)[0]),str(g.kernel_),float(g.log_marginal_likelihood_value_)]
A['30B.3']={k_:[round(v[0],3),v[1],round(v[2],3)] for k_,v in res.items()}
fig,ax=small(); xg=np.linspace(0,35,200)
for flag,c,ls_ in [(False,RED,'--'),(True,GREEN,'-')]:
    g=GPR(C(1.0)*RBF(1.5),alpha=0.01,normalize_y=flag,random_state=0,n_restarts_optimizer=3).fit(TX[:,None],Y100); ax.plot(xg,g.predict(xg[:,None]),color=c,lw=1.6,ls=ls_)
ax.scatter(TX,Y100,s=18,color=BLUE,edgecolor='white',lw=0.6,zorder=6); ax.set_ylim(-8,110); ax.set_xlabel('$x$'); ax.set_ylabel('mean prediction'); grd(ax)
ax.text(34.5,38,'normalize_y=False',color=RED,fontsize=9,ha='right'); ax.text(34.5,58,'normalize_y=True',color=GREEN,fontsize=9,ha='right'); sv(fig,'p_30B3.png')
# ---- 30C.1 the band ignores y
fig,axs=plt.subplots(1,2,figsize=(2.9,1.25),dpi=240,sharey=True); xs3=np.array([1.,2,3,4,5,6]); ya=np.array([0.3,0.6,0.9,1.0,0.8,0.5]); yb=np.array([0.6,-0.6,0.6,-0.6,0.6,-0.6])
for ax,yy,tt in zip(axs,[ya,yb],['smooth data','zigzag data']):
    g=GPR(C(1.0,'fixed')*RBF(1.5,'fixed'),alpha=0.01,optimizer=None).fit(xs3[:,None],yy); gg=np.linspace(0,7,100); m_,s_=g.predict(gg[:,None],return_std=True)
    ax.fill_between(gg,m_-2*s_,m_+2*s_,color=LG,lw=0); ax.plot(gg,m_,color=GREEN,lw=1.4); ax.scatter(xs3,yy,s=14,color=BLUE,edgecolor='white',lw=0.5,zorder=6); ax.set_title(tt,fontsize=9,color=NAVY,pad=2); ax.set_xticks([0,3,6]); ax.set_ylim(-3,3); ax.set_yticks([-2,0,2]); grd(ax); ax.tick_params(labelsize=9)
    A.setdefault('30C.1',{})[tt]=round(float(s_[np.argmin(abs(gg-3.5))]),4)
fig.tight_layout(w_pad=0.4); sv(fig,'p_30C1.png')
# ---- 30C.2 Matern or RBF on a kink
from sklearn.gaussian_process.kernels import Matern
xk=np.linspace(-3,3,9); yk=np.abs(xk)+0.05*np.random.default_rng(1).normal(size=9); gg=np.linspace(-4,4,200)
fig,ax=small(2.4,0.95)
for kern,c,ls_,lab in [(RBF(1.0),RED,'--','RBF'),(Matern(1.0,nu=0.5),GREEN,'-','Matérn $\\nu=\\frac{1}{2}$')]:
    g=GPR(C(1.0)*kern,alpha=0.01,normalize_y=True,random_state=0,n_restarts_optimizer=3).fit(xk[:,None],yk); ax.plot(gg,g.predict(gg[:,None]),color=c,lw=1.5,ls=ls_,label=lab); A.setdefault('30C.2',{})[lab]=[str(g.kernel_),round(float(g.log_marginal_likelihood_value_),3)]
ax.scatter(xk,yk,s=16,color=BLUE,edgecolor='white',lw=0.5,zorder=6); ax.legend(loc='upper center',fontsize=9,labelspacing=0.2,handlelength=1.6,borderaxespad=0.1); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_ylim(-0.5,4.4); grd(ax); sv(fig,'p_30C2.png')
# ---- 30C.3 cost versus n
ck=json.load(open('verify/checks.json')); ns=[250,500,1000,2000]; ts=[ck[f't{n}'] for n in ns]
fig,ax=small(); ax.loglog(ns,ts,'o-',color=BLUE,lw=1.4,ms=4,label='measured'); ref=np.array(ns,float); ax.loglog(ns,ts[-1]*(ref/ns[-1])**3,color=PUR,lw=1.0,ls='--',label='$n^3$')
ax.set_xlabel('training samples $n$'); ax.set_ylabel('fit time (s)'); ax.legend(fontsize=9,loc='upper left',handlelength=1.6,borderaxespad=0.1); grd(ax); ax.set_xticks(ns); ax.set_xticklabels(['250','500','1000','2000']); ax.minorticks_off(); sv(fig,'p_30C3.png')
A['30C.3']={'mem_GB_n20000':8*20000**2/1e9}
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/prob_fontcheck.json','w'),indent=1); print(json.dumps(A,indent=0)); print(CHK)

# All figures of Chapter 5. Run: python3 fig/figs.py   (PNGs into ch05/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Ellipse
NUM={}
x=np.arange(1.,6); y=np.array([3.,4,6,7,10]); X=np.c_[np.ones(5),x]; H=2*X.T@X
Jd=lambda b,w:((y[None,None,:]-b[...,None]-w[...,None]*x[None,None,:])**2).sum(-1)
def grad(v): return -2*X.T@(y-X@v)
def gd_path(v0,eta,n,Xm=X,ym=y):
    v=np.array(v0,float); P=[v.copy()]
    for _ in range(n):
        v=v-eta*(-2*Xm.T@(ym-Xm@v)); P.append(v.copy())
        if np.abs(v).max()>1e3: break
    return np.array(P)
def cd_path(v0,n,Xm=X,ym=y):
    v=np.array(v0,float); P=[v.copy()]
    for _ in range(n):
        for j in range(Xm.shape[1]):
            xj=Xm[:,j]; r=ym-Xm@v+xj*v[j]; v[j]=xj@r/(xj@xj); P.append(v.copy())
    return np.array(P)
def grid(b0,b1,w0,w1,n=300):
    bb=np.linspace(b0,b1,n); ww=np.linspace(w0,w1,n); return np.meshgrid(bb,ww)
LV=[1.1,2,4,8,16,32,64,128,256,512]
def contours(ax,Bg,Wg,Z,levels=LV,lw=0.7,fill=True):
    if fill: ax.contourf(Bg,Wg,Z,levels=levels+[1e9],colors=None,cmap='Blues_r',alpha=0.55,norm=matplotlib.colors.LogNorm(vmin=0.5,vmax=2000))
    ax.contour(Bg,Wg,Z,levels=levels,colors=BLUE,linewidths=lw,alpha=0.8)
def star(ax,p,ms=12): ax.plot(*p,'*',color=GREEN,ms=ms,mec='white',mew=0.7,zorder=7)

# ---- 5.0-1 opener: three methods on the delivery bowl
fig,ax=plt.subplots(figsize=(5.4,2.15),dpi=240)
Bg,Wg=grid(-1,7.2,-0.2,3.2); contours(ax,Bg,Wg,Jd(Bg,Wg))
lam=np.linalg.eigvalsh(H); eta=1/lam[-1]
P=gd_path([0,0],eta,60); ax.plot(P[:,0],P[:,1],'-o',color=NAVY,ms=2.8,lw=1.0,zorder=5,label='gradient descent (60 steps)')
Pc=cd_path([0,0],6); ax.plot(Pc[:,0],Pc[:,1],'-s',color=OR,ms=3,lw=1.0,zorder=5,label='coordinate descent (6 sweeps)')
d=np.linalg.solve(H,grad(np.zeros(2))); ax.annotate('',xy=(0,0)-d,xytext=(0,0),arrowprops=dict(arrowstyle='-|>',color=PUR,lw=1.6,ls='--',mutation_scale=10),zorder=6)
ax.plot([],[],'--',color=PUR,lw=1.6,label='Newton (1 step)')
ax.plot(0,0,'D',color='white',mec=NAVY,ms=6,zorder=8); ax.text(0.15,-0.12,'start',fontsize=9,color=NAVY,va='top')
star(ax,(0.9,1.7)); ax.text(1.2,1.62,'minimum',fontsize=9,color=GREEN,weight='bold',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.8))
ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); ax.set_xlim(-1,7.2); ax.set_ylim(-0.2,3.2)
ax.legend(loc='upper right',fontsize=9,frameon=True,facecolor='white',edgecolor=GRID,framealpha=0.95)
save(fig,'f500.png'); NUM['eta_1_over_lmax']=float(eta)

# ---- 5.2-1 convex and non-convex sets
fig,axs=plt.subplots(1,2,figsize=(4.4,1.9),dpi=240)
th=np.linspace(0,2*np.pi,200)
ax=axs[0]; ax.fill(np.cos(th)*1.0,0.75*np.sin(th),fc=LB,ec=BLUE,lw=1.0); a,b_=np.array([-0.7,-0.3]),np.array([0.65,0.35]); ax.plot(*zip(a,b_),color=GREEN,lw=1.6); ax.plot(*zip(a,b_),'o',color=GREEN,ms=4)
ax.set_title('convex set',fontsize=10,color=NAVY); ax.text(0,-1.0,'segment stays inside',ha='center',fontsize=9,color=GREEN)
ax=axs[1]; L=np.array([[-1,-0.75],[1,-0.75],[1,-0.15],[-0.35,-0.15],[-0.35,0.75],[-1,0.75]]); ax.fill(L[:,0],L[:,1],fc=LB,ec=BLUE,lw=1.0)
a,b_=np.array([0.85,-0.45]),np.array([-0.7,0.55]); ax.plot(*zip(a,b_),color=RED,lw=1.6,zorder=4); ax.plot(*zip(a,b_),'o',color=RED,ms=4,zorder=5)
ax.set_title('not convex',fontsize=10,color=NAVY); ax.text(0,-1.0,'segment leaves the set',ha='center',fontsize=9,color=RED)
for ax in axs: ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-1.25,1.25); ax.set_ylim(-1.2,1.0)
fig.tight_layout(w_pad=0.5); save(fig,'f521.png')

# ---- 5.2-2 convex function with chord, non-convex with two minima, three losses
fig,axs=plt.subplots(1,3,figsize=(5.7,2.15),dpi=240)
ax=axs[0]; z=np.linspace(-1.6,1.9,200); f=lambda z:0.6*z**2; ax.plot(z,f(z),color=BLUE,lw=1.6)
a_,b_=-1.2,1.5; ax.plot([a_,b_],[f(a_),f(b_)],color=GREEN,lw=1.4); ax.plot([a_,b_],[f(a_),f(b_)],'o',color=GREEN,ms=4)
m=0.5*(a_+b_); ax.plot([m,m],[f(m),0.5*(f(a_)+f(b_))],color=RED,lw=1.2); ax.plot(m,f(m),'o',color=BLUE,ms=4); ax.plot(m,0.5*(f(a_)+f(b_)),'o',color=GREEN,ms=4)
ax.set_title('convex: chord above',fontsize=9,color=NAVY); ax.text(-1.55,1.85,'one minimum',fontsize=9,color=GREEN,ha='left')
ax=axs[1]; z=np.linspace(-2.3,2.3,300); f2=lambda z:0.25*z**4-0.9*z**2+0.35*z; ax.plot(z,f2(z),color=BLUE,lw=1.6)
zs=np.linspace(-2.3,2.3,4601); fz=f2(zs); ig=np.argmin(fz); zg=zs[ig]
msk=zs>0.2; il=np.argmin(np.where(msk,fz,1e9)); zl=zs[il]
ax.plot(zg,f2(zg),'*',color=GREEN,ms=11,mec='white',mew=0.6); ax.plot(zl,f2(zl),'o',color=OR,ms=6,mec='white',mew=0.6)
ax.text(zg,f2(zg)-0.55,'global',ha='center',fontsize=9,color=GREEN); ax.text(zl+0.1,f2(zl)-0.55,'local',ha='center',fontsize=9,color=OR)
ax.set_ylim(-3.1,1.8); ax.set_title('not convex',fontsize=9,color=NAVY)
NUM['nonconvex']={'z_global':round(float(zg),3),'f_global':round(float(f2(zg)),3),'z_local':round(float(zl),3),'f_local':round(float(f2(zl)),3)}
ax=axs[2]; r=np.linspace(-2.2,2.2,300)
ax.plot(r,r**2,color=BLUE,lw=1.5,label='squared'); ax.plot(r,np.abs(r),color=OR,lw=1.5,ls='--',label='absolute'); ax.plot(r,np.logaddexp(0,-2*r),color=RED,lw=1.5,ls=':',label='log loss')
ax.set_ylim(-0.2,4.3); ax.legend(fontsize=9,loc='upper center',handlelength=1.8,borderaxespad=0.1,labelspacing=0.25); ax.set_title('three convex losses',fontsize=9,color=NAVY)
for ax in axs: ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
axs[0].set_xlabel('$z$'); axs[1].set_xlabel('$z$'); axs[2].set_xlabel('error or margin')
axs[0].set_ylim(-0.15,2.2)
fig.tight_layout(w_pad=0.7); save(fig,'f522.png')

# ---- 5.3-1 gradient descent paths for three learning rates
fig,axs=plt.subplots(1,3,figsize=(5.7,2.5),dpi=240,sharey=True)
Bg,Wg=grid(-1.5,7,-0.3,3.3); Zd=Jd(Bg,Wg); NUM['gd_paths']={}
for ax,(eta_,tt,nst) in zip(axs,[(0.002,'$\\eta$ = 0.002: too small',60),(1/lam[-1],'$\\eta$ = 0.0085: good',60),(0.0172,'$\\eta$ = 0.0172: too large',9)]):
    contours(ax,Bg,Wg,Zd,lw=0.6); P=gd_path([0,0],eta_,nst)
    ax.plot(P[:,0],P[:,1],'-o',color=NAVY,ms=2.6,lw=0.9,zorder=5); ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8); star(ax,(0.9,1.7),10)
    ax.set_title(tt,fontsize=9,color=NAVY); ax.set_xlim(-1.5,7); ax.set_ylim(-0.3,3.3); ax.set_xlabel('intercept $b$')
    JP=Jd(P[:,0:1],P[:,1:2])[:,0]; NUM['gd_paths'][str(round(eta_,4))]={'steps':len(P)-1,'J_end':float(JP[-1]),'J_first3':[float(v) for v in JP[:4]]}
axs[0].set_ylabel('slope $w$')
axs[1].text(6.9,2.9,'steep walls first,\nthen a slow crawl',fontsize=9,color=NAVY,ha='right',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
axs[2].text(6.8,2.9,'swings grow\nat every step',fontsize=9,color=RED,ha='right',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
fig.tight_layout(w_pad=0.5); save(fig,'f531.png')

# ---- 5.3-2 feature scaling: elongated bowls
import importlib
sq2=np.sqrt(2)
designs=[('raw $x$',X,(-1,7.2,-0.2,3.2)),('centred $x$',np.c_[np.ones(5),x-3],None),('standardised $x$',np.c_[np.ones(5),(x-3)/sq2],None)]
def steps_to_tol(Xm,eta,tol=1e-6):
    v=np.zeros(2)
    for t in range(200000):
        g=-2*Xm.T@(y-Xm@v)
        if np.linalg.norm(g)<tol: return t
        v=v-eta*g
    return None
fig,axs=plt.subplots(1,3,figsize=(5.7,2.2),dpi=240); NUM['scaling']={}
for ax,(nm,Xm,lim),wopt in zip(axs,designs,[1.7,1.7,1.7*sq2]):
    Hm=2*Xm.T@Xm; lm=np.linalg.eigvalsh(Hm); et=1/lm[-1]; n=steps_to_tol(Xm,et); kap=lm[-1]/lm[0]
    vopt=np.linalg.solve(Xm.T@Xm,Xm.T@y)
    cb=vopt[0]; cw=vopt[1]
    if nm.startswith('raw'): b0,b1,w0,w1=-2,7,-0.5,3.5
    else: b0,b1,w0,w1=-1.5,8.5,-0.8,cw+1.9
    Bg,Wg=grid(b0,b1,w0,w1); Z=((y[None,None,:]-Xm[:,0][None,None,:]*Bg[...,None]-Xm[:,1][None,None,:]*Wg[...,None])**2).sum(-1)
    lv=np.sort(np.array([0.5,1,2,4,8,16,32,64,128,256,512])*Z.min()+ (Z.min() if Z.min()>0 else 0)) if False else [1.1*k for k in (1,1.6,2.5,4,8,16,32,64,128,256)]
    ax.contour(Bg,Wg,Z,levels=lv,colors=BLUE,linewidths=0.7); ax.contourf(Bg,Wg,Z,levels=lv+[1e9],cmap='Blues_r',alpha=0.45,norm=matplotlib.colors.LogNorm(vmin=0.5,vmax=2000))
    P=gd_path([0,0],et,40,Xm,y); ax.plot(P[:,0],P[:,1],'-o',color=NAVY,ms=2.4,lw=0.9,zorder=5); ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8)
    star(ax,(cb,cw),10); ax.set_xlim(b0,b1); ax.set_ylim(w0,w1)
    ax.set_title(nm,fontsize=9.5,color=NAVY); ax.set_xlabel('intercept'); 
    ax.text(0.97,0.95,f'cond. {kap:.0f}\n{n} step'+('s' if n!=1 else ''),transform=ax.transAxes,fontsize=9,color=RED,weight='bold',va='top',ha='right',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
    NUM['scaling'][nm]={'cond':float(kap),'steps':n,'eta':float(et)}
axs[0].set_ylabel('slope')
fig.tight_layout(w_pad=0.6); save(fig,'f532.png')

# ---- 5.4-1 SGD: noisy paths and noise floor
from sklearn.linear_model import LinearRegression
rng=np.random.default_rng(0); n=100; xs=rng.normal(size=n); ys=2+3*xs+rng.normal(0,1.5,n); Xs=np.c_[np.ones(n),xs]
vopt=np.linalg.lstsq(Xs,ys,rcond=None)[0]; Jmin=0.5*np.mean((ys-Xs@vopt)**2)
Jfun=lambda v:0.5*np.mean((ys-Xs@v)**2)
def sgd_run(batch,eta_fn,epochs,seed=0):
    rg=np.random.default_rng(seed); v=np.zeros(2); P=[v.copy()]; E=[Jfun(v)-Jmin]; t=0
    for e in range(epochs):
        idx=rg.permutation(n)
        for k in range(0,n,batch):
            bi=idx[k:k+batch]; g=-Xs[bi].T@(ys[bi]-Xs[bi]@v)/len(bi); v=v-eta_fn(e,t)*g; t+=1; P.append(v.copy())
        E.append(Jfun(v)-Jmin)
    return np.array(P),np.array(E)
c01=lambda e,t:0.1
runs={'full batch (100)':sgd_run(100,c01,30),'mini-batch 10':sgd_run(10,c01,30),'one sample':sgd_run(1,c01,30),'one sample, decaying $\\eta$':sgd_run(1,lambda e,t:0.1/(1+0.3*e),30)}
fig,axs=plt.subplots(1,2,figsize=(5.7,2.7),dpi=240)
ax=axs[0]; Bg,Wg=grid(-0.6,2.8,-0.6,3.6); Zs=0.5*np.mean((ys[None,None,:]-Bg[...,None]-Wg[...,None]*xs[None,None,:])**2,axis=-1)
ax.contour(Bg,Wg,Zs,levels=Jmin+np.array([0.05,0.2,0.5,1,2,3.5,5.5]),colors=BLUE,linewidths=0.6,alpha=0.8)
sty={'full batch (100)':('-o',NAVY,3),'mini-batch 10':('-s',OR,2.5),'one sample':('-',RED,0)}
for k,(nm,(P,Ee)) in enumerate(runs.items()):
    if nm in sty:
        st,c_,ms=sty[nm]; m_=40 if nm!='full batch (100)' else 40
        ax.plot(P[:m_+1,0],P[:m_+1,1],st,color=c_,ms=ms,lw=0.9,label=nm if nm!='full batch (100)' else 'full batch',zorder=5)
ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8); star(ax,vopt,10); ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); ax.set_xlim(-0.6,2.8); ax.set_ylim(-0.6,3.6)
ax.legend(fontsize=9,loc='lower right',frameon=True,facecolor='white',edgecolor=GRID,framealpha=0.95,title='first 40 updates',title_fontsize=9)
ax=axs[1]; cs={'full batch (100)':NAVY,'mini-batch 10':OR,'one sample':RED,'one sample, decaying $\\eta$':PUR}; ls={'full batch (100)':'-','mini-batch 10':'--','one sample':'-.','one sample, decaying $\\eta$':':'}
for nm,(P,Ee) in runs.items(): ax.semilogy(np.arange(len(Ee)),Ee,ls[nm],color=cs[nm],lw=1.5,label=nm if nm!='full batch (100)' else 'full batch')
ax.set_xlabel('epochs (passes over the data)'); ax.set_ylabel('loss above the minimum'); ax.grid(color=GRID,lw=0.5); ax.set_ylim(1e-5,3e3); ax.legend(fontsize=9,loc='upper right')
fig.tight_layout(w_pad=1.0); save(fig,'f541.png')
NUM['sgd']={nm:{'final_excess':float(Ee[-1]),'mean_last10':float(np.mean(Ee[-10:])),'n_updates':len(P)-1} for nm,(P,Ee) in runs.items()}
NUM['sgd_opt']=[float(v) for v in vopt]

# ---- 5.4-2 schedules; momentum and Adam
fig,axs=plt.subplots(1,2,figsize=(5.7,2.6),dpi=240)
ax=axs[0]; t=np.arange(1,201)
ax.plot(t,[0.01]*200,color=NAVY,lw=1.5,label="constant"); ax.plot(t,0.01/t**0.25,color=OR,lw=1.5,ls='--',label="invscaling, $p$ = 0.25"); ax.plot(t,0.01/t**0.5,color=RED,lw=1.5,ls='-.',label="invscaling, $p$ = 0.5")
ad=np.where(t<60,0.01,np.where(t<120,0.002,0.0004)); ax.plot(t,ad,color=PUR,lw=1.5,ls=':',label='adaptive (schematic)')
ax.set_xlabel('update number $t$'); ax.set_ylabel('learning rate $\\eta_t$'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=9,loc='upper right',labelspacing=0.3); ax.set_ylim(0,0.0165)
ax=axs[1]; Bg,Wg=grid(-0.6,3.2,-0.2,3.5); Zd2=Jd(Bg,Wg); ax.contour(Bg,Wg,Zd2,levels=LV,colors=BLUE,linewidths=0.6,alpha=0.8)
eta=1/lam[-1]
def heavy(v0,eta,mu,n):
    v=np.array(v0,float); u=np.zeros(2); P=[v.copy()]
    for _ in range(n): u=mu*u-eta*grad(v); v=v+u; P.append(v.copy())
    return np.array(P)
def adam(v0,lr,n,b1=0.9,b2=0.999,eps=1e-8):
    v=np.array(v0,float); m=np.zeros(2); s=np.zeros(2); P=[v.copy()]
    for t_ in range(1,n+1):
        g=grad(v); m=b1*m+(1-b1)*g; s=b2*s+(1-b2)*g**2; l=lr*np.sqrt(1-b2**t_)/(1-b1**t_); v=v-l*m/(np.sqrt(s)+eps); P.append(v.copy())
    return np.array(P)
Pg=gd_path([0,0],eta,40); Pm=heavy([0,0],0.008,0.8,40); Pa=adam([0,0],0.15,40)
for P,st,c_,nm in [(Pg,'-o',NAVY,'plain'),(Pm,'-s',OR,'momentum'),(Pa,'-^',PUR,'Adam')]: ax.plot(P[:,0],P[:,1],st,color=c_,ms=2.5,lw=0.9,label=nm,zorder=5)
ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8); star(ax,(0.9,1.7),10); ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); ax.set_xlim(-0.6,3.2); ax.set_ylim(-0.2,3.4); ax.legend(fontsize=9,loc='lower right',frameon=True,facecolor='white',edgecolor=GRID,framealpha=0.95)
fig.tight_layout(w_pad=1.0); save(fig,'f542.png')
def steps_within(P,tol=0.01): 
    Jv=Jd(P[:,0:1],P[:,1:2])[:,0]; k=np.where(Jv<1.1+tol)[0]; return int(k[0]) if len(k) else None
def run_until(fn,tol=0.01,n=5000):
    for k in range(n):
        pass
NUM['momentum']={'plain':steps_within(gd_path([0,0],eta,3000),0.01),'heavy_ball':steps_within(heavy([0,0],0.008,0.8,3000),0.01),'adam':steps_within(adam([0,0],0.15,3000),0.01)}

# ---- 5.5-1 Newton in one variable
sig=lambda w:1/(1+np.exp(-w)); f1=lambda w:np.logaddexp(0,w)-0.3*w; d1=lambda w:sig(w)-0.3; d2=lambda w:sig(w)*(1-sig(w)); wst=np.log(0.3/0.7)
fig,axs=plt.subplots(1,2,figsize=(5.7,2.5),dpi=240)
ax=axs[0]; w=np.linspace(-2.6,2.1,300); ax.plot(w,f1(w),color=BLUE,lw=1.8,label='$f(w)$')
w0=0.0; q=lambda w,w0:f1(w0)+d1(w0)*(w-w0)+0.5*d2(w0)*(w-w0)**2
ax.plot(w,q(w,w0),color=PUR,lw=1.2,ls='--',label='parabola at $w_0$'); w1=w0-d1(w0)/d2(w0)
ax.plot(w,q(w,w1),color=OR,lw=1.2,ls=':',label='parabola at $w_1$'); w2=w1-d1(w1)/d2(w1)
for k,(wk,c_) in enumerate([(w0,PUR),(w1,OR)]):
    ax.plot(wk,f1(wk),'o',color=c_,ms=5,zorder=5); ax.text(wk+(0.1 if k==0 else -0.1),f1(wk)+0.08,f'$w_{k}$',color=c_,ha='left' if k==0 else 'right',va='bottom',fontsize=10)
ax.plot(wst,f1(wst),'*',color=GREEN,ms=11,mec='white',zorder=6); ax.annotate('',xy=(w1,q(w1,w0)),xytext=(w0,f1(w0)),arrowprops=dict(arrowstyle='-|>',color=PUR,lw=1.0,connectionstyle='arc3,rad=0.25'))
ax.set_ylim(0.52,1.55); ax.set_xlim(-2.6,2.1); ax.set_xlabel('$w$'); ax.set_ylabel('$f(w)$'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=9,loc='upper center',handlelength=1.8,ncol=1,labelspacing=0.25)
NUM['newton1d']={'w1':float(w1),'w2':float(w2)}
ax=axs[1]; wn=wg=0.0; En=[abs(wn-wst)]; Eg=[abs(wg-wst)]
for t_ in range(6): wn=wn-d1(wn)/d2(wn); wg=wg-d1(wg); En.append(max(abs(wn-wst),1e-16)); Eg.append(abs(wg-wst))
ax.semilogy(range(7),En,'-s',color=PUR,ms=4,lw=1.4,label='Newton'); ax.semilogy(range(7),Eg,'o',color=NAVY,ms=4,lw=1.4,ls='--',label='gradient descent, $\\eta$ = 1')
ax.set_xlabel('iteration $t$'); ax.set_ylabel('error $|w_t - w^{\\ast}|$'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=9,loc='center left',bbox_to_anchor=(0.0,0.3),handlelength=1.8); ax.set_ylim(1e-17,3)
fig.tight_layout(w_pad=1.0); save(fig,'f551.png')

# ---- 5.6-1 coordinate descent
fig,axs=plt.subplots(1,2,figsize=(5.7,2.5),dpi=240)
ax=axs[0]; Bg,Wg=grid(-0.5,7,-0.3,3.2); contours(ax,Bg,Wg,Jd(Bg,Wg),lw=0.6); Pc=cd_path([0,0],8)
ax.plot(Pc[:,0],Pc[:,1],'-s',color=OR,ms=2.8,lw=1.0,zorder=5); ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8); star(ax,(0.9,1.7),10)
ax.set_title('raw $x$: slow staircase',fontsize=9.5,color=NAVY); ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); ax.set_xlim(-0.5,7); ax.set_ylim(-0.3,3.2)
ax.text(6.9,2.9,'first 8 sweeps\n(94 needed)',fontsize=9,color=OR,ha='right',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
ax=axs[1]; Xc=np.c_[np.ones(5),x-3]; Bg,Wg=grid(-0.5,7,-0.3,3.2)
Zc=((y[None,None,:]-Bg[...,None]-Wg[...,None]*(x-3)[None,None,:])**2).sum(-1); contours(ax,Bg,Wg,Zc,lw=0.6,levels=[1.1*k for k in (1,1.5,3,6,12,24,48,96,192)])
Pc=cd_path([0,0],2,Xc,y); ax.plot(Pc[:,0],Pc[:,1],'-s',color=OR,ms=3.5,lw=1.2,zorder=5); ax.plot(0,0,'D',color='white',mec=NAVY,ms=5,zorder=8); star(ax,(6,1.7),10)
ax.set_title('centred $x$: one sweep',fontsize=9.5,color=NAVY); ax.set_xlabel('intercept $b$'); ax.set_xlim(-0.5,7); ax.set_ylim(-0.3,3.2)
fig.tight_layout(w_pad=0.8); save(fig,'f561.png')

# ---- 5.7-1 Lagrange geometry
fig,axs=plt.subplots(1,2,figsize=(5.7,2.9),dpi=240)
c=np.array([3.,4.]); th=np.linspace(0,2*np.pi,400)
def base(ax):
    for r_ in (1,2,3,4,5): ax.plot(c[0]+r_*np.cos(th),c[1]+r_*np.sin(th),color=BLUE,lw=0.5,alpha=0.35)
    ax.plot(*c,'o',color=BLUE,ms=4,zorder=6); ax.text(c[0]+0.15,c[1]+0.1,'unconstrained\nminimum $(3,4)$',fontsize=9,color=BLUE,va='bottom')
    ax.axhline(0,color=GRID,lw=0.6); ax.axvline(0,color=GRID,lw=0.6); ax.set_aspect('equal'); ax.set_xlim(-1.5,6.6); ax.set_ylim(-1.5,5.9)
ax=axs[0]; base(ax); ax.plot(np.cos(th),np.sin(th),color=RED,lw=1.8); ax.fill(np.cos(th),np.sin(th),color=PINK,alpha=0.8,zorder=1)
ws=np.array([0.6,0.8]); rr=np.linalg.norm(c-ws); ax.plot(c[0]+rr*np.cos(th),c[1]+rr*np.sin(th),color=BLUE,lw=1.5)
ax.plot(*ws,'*',color=GREEN,ms=12,mec='white',zorder=8)
nrm=np.array([0.6,0.8]); per=np.array([-0.8,0.6])*0.1
arr(ax,tuple(ws+per),tuple(ws+per+2.3*nrm),c=GREEN,lw=1.5,ms=10); arr(ax,tuple(ws-per),tuple(ws-per+1.0*nrm),c=RED,lw=1.5,ms=10)
ax.text(2.05,2.0,'$-\\nabla f$',color=GREEN,fontsize=11); ax.text(-0.45,1.75,'$\\nabla g$',color=RED,fontsize=11,ha='right')
ax.text(6.0,-1.2,'solution $(0.6,\\,0.8)$',fontsize=9.5,color=NAVY,ha='right',bbox=dict(fc='white',ec='none',pad=1))
ax.set_title('L2 ball: the contour touches the circle',fontsize=9.5,color=NAVY)
ax=axs[1]; base(ax); dx=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]]); ax.fill(dx[:,0],dx[:,1],color=PINK,alpha=0.8,zorder=1); ax.plot(dx[:,0],dx[:,1],color=RED,lw=1.8)
ws=np.array([0.,1.]); rr=np.linalg.norm(c-ws); ax.plot(c[0]+rr*np.cos(th),c[1]+rr*np.sin(th),color=BLUE,lw=1.5); ax.plot(*ws,'*',color=GREEN,ms=12,mec='white',zorder=8)
arr(ax,tuple(ws),tuple(ws+0.4*(c-ws)/np.linalg.norm(c-ws)*3),c=GREEN,lw=1.4,ms=9)
ax.text(6.0,-1.2,'solution $(0,\\,1)$',fontsize=9.5,color=NAVY,ha='right',bbox=dict(fc='white',ec='none',pad=1)); ax.set_title('L1 ball: the contour touches a corner',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=0.4); save(fig,'f571.png')

# ---- 5.9-1 flow chart
fig,ax=plt.subplots(figsize=(5.4,4.3),dpi=240); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(0,8.4)
def rbox(x0,y0,w,h,txt,fc,ec,fs=9.5): 
    ax.add_patch(FancyBboxPatch((x0,y0),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec=ec,lw=1.0)); ax.text(x0+w/2,y0+h/2,txt,ha='center',va='center',fontsize=fs,color=NAVY)
def dia(cx,cy,w,h,txt,fs=9.5):
    ax.add_patch(Polygon([[cx-w/2,cy],[cx,cy+h/2],[cx+w/2,cy],[cx,cy-h/2]],fc='#FDF0E0',ec=OR,lw=1.0)); ax.text(cx,cy,txt,ha='center',va='center',fontsize=fs,color=NAVY)
def A(p,q): arr(ax,p,q,c=NAVY,lw=1.0,ms=8)
rbox(1.2,7.3,5.6,0.95,'ConvergenceWarning, or\nn_iter_ equals max_iter',LGREY,GREY)
dia(4.0,5.85,5.4,1.6,'features on very\ndifferent scales?'); A((4.0,7.3),(4.0,6.65))
rbox(7.45,5.4,3.45,0.9,'StandardScaler in a\nPipeline, then refit',LB,BLUE); A((6.7,5.85),(7.45,5.85)); ax.text(7.07,6.0,'yes',fontsize=9,color=NAVY,ha='center')
dia(4.0,3.75,5.4,1.6,'warning gone after\n10 times max_iter?'); A((4.0,5.05),(4.0,4.55)); ax.text(4.15,4.8,'no',fontsize=9,color=NAVY,ha='left')
rbox(7.45,3.3,3.45,0.9,'keep it; record\nn_iter_',  '#E3F4EA',GREEN); A((6.7,3.75),(7.45,3.75)); ax.text(7.07,3.9,'yes',fontsize=9,color=NAVY,ha='center')
ax.plot([4.0,4.0],[2.95,2.45],color=NAVY,lw=1.0); ax.text(4.15,2.7,'no',fontsize=9,color=NAVY,ha='left'); ax.plot([2.6,8.3],[2.45,2.45],color=NAVY,lw=1.0)
A((2.6,2.45),(2.6,2.0)); A((8.3,2.45),(8.3,2.0))
rbox(0.15,0.4,5.0,1.6,'try another solver, or\nstronger regularisation\n(smaller C, larger alpha)',LB,BLUE)
rbox(5.85,0.4,5.0,1.6,'inspect the data: duplicated\ncolumns, separable classes,\nhuge or constant columns','#F1EAF7',PUR)
save(fig,'f591.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

# All figures of Chapter 3. Run: python3 fig/figs.py   (PNGs into ch03/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
NUM={}

def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def title(ax,t): ax.set_title(t,fontsize=9.5,color=NAVY)
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10])
Xd=np.c_[np.ones(5),x]
Jd=lambda b,w: ((y[None,None,:]-b[...,None]-w[...,None]*x[None,None,:])**2).sum(-1)
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8})
# ---- 3.0-1 opener: slope in one variable, gradient in two
fig,axs=plt.subplots(1,2,figsize=(4.31,1.91),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; f=lambda w: 0.6*(w-2)**2+0.5; df=lambda w: 1.2*(w-2); ww=np.linspace(-0.3,4.3,200); ax.plot(ww,f(ww),color=BLUE,lw=2)
NUM['open_slopes']=[]
for w0,lab,dy in [(0.4,'slope',0.0),(2.0,'slope',0),(3.5,'slope',0)]:
    s=df(w0); t=np.array([w0-0.65,w0+0.65]); ax.plot(t,f(w0)+s*(t-w0),color=GREEN,lw=1.4); ax.plot(w0,f(w0),'o',color=RED,ms=5,zorder=5); NUM['open_slopes'].append(round(float(s),2))
ax.text(0.4,-0.1,f'slope \u2212{abs(df(0.4)):.1f}\nmove right',fontsize=8.5,color=RED,ha='center',va='top')
ax.text(2.0,f(2.0)-0.0,'',fontsize=8.5)
ax.text(2.0,-0.1,'slope 0\nminimum',fontsize=8.5,color=RED,ha='center',va='top')
ax.text(3.6,-0.1,f'slope +{df(3.5):.1f}\nmove left',fontsize=8.5,color=RED,ha='center',va='top')
ax.set_xlim(-0.4,4.5); ax.set_ylim(-1.5,4.2); ax.set_xlabel('parameter $w$'); ax.set_ylabel('loss $J$'); grid(ax); title(ax,'One parameter: slope')
ax=axs[1]; bx_=np.linspace(-1.3,1.9,300); by_=np.linspace(-0.7,2.3,300); Bg,Yg=np.meshgrid(bx_,by_); F=lambda u,v:(1-u)**2+3*(v-u**2)**2
ax.contour(Bg,Yg,F(Bg,Yg),levels=[0.05,0.2,0.6,1.5,3,6,12],colors=BLUE,linewidths=0.9,alpha=0.85); ax.set_aspect('equal')
ax.plot(1,1,'*',color=GREEN,ms=13,mec='white',zorder=6)
def g(u,v): return np.array([-2*(1-u)-12*u*(v-u**2),6*(v-u**2)])
for p in [(-1.0,0.3),(-0.5,1.5),(0.4,0.0),(1.3,0.4),(0.9,1.9),(0.0,1.1)]:
    gg=g(*p); gg=0.5*gg/np.linalg.norm(gg); arr(ax,p,(p[0]+gg[0],p[1]+gg[1]),c=RED,lw=1.4,ms=8); ax.plot(*p,'o',color=NAVY,ms=3,zorder=5)
ax.text(0.93,1.12,'minimum',fontsize=8.5,color=GREEN,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1,alpha=0.9))
ax.set_xlim(-1.3,1.9); ax.set_ylim(-0.7,2.3); ax.set_xlabel('parameter $w_1$'); ax.set_ylabel('parameter $w_2$'); title(ax,'Two parameters: gradient (red)')
fig.tight_layout(w_pad=1.2); save(fig,'f300.png')

# ---- 3.1-1 secants to tangent, and the derivative as a function
fig,axs=plt.subplots(1,2,figsize=(4.31,1.95),dpi=240); ax=axs[0]; f=lambda t:t**2; tt=np.linspace(-0.2,3.2,200); ax.plot(tt,f(tt),color=BLUE,lw=2)
NUM['secant']={}
for h,ls in [(1.0,'-'),(0.5,'--'),(0.2,':')]:
    s=(f(1+h)-f(1))/h; NUM['secant'][str(h)]=round(float(s),2); xs=np.array([0.3,3.0]); ax.plot(xs,f(1)+s*(xs-1),color=PUR,lw=1.3,ls=ls); ax.plot(1+h,f(1+h),'o',color=PUR,ms=4,zorder=5)
xs=np.array([0.3,2.2]); ax.plot(xs,1+2*(xs-1),color=GREEN,lw=1.8); ax.plot(1,1,'o',color=RED,ms=6,zorder=6)
ax.text(0.15,8.3,'secant slopes (purple):\n3.0, 2.5, 2.2 for\n$h$ = 1, 0.5, 0.2',fontsize=8,color=PUR,va='top'); ax.text(2.25,2.0,'tangent: slope 2',fontsize=8.5,color=GREEN,va='top',ha='left')
ax.annotate('$x$ = 1',xy=(1,1),xytext=(0.0,2.3),fontsize=8.5,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.6))
ax.set_xlim(-0.2,3.2); ax.set_ylim(-0.3,9.6); ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)=x^2$'); grid(ax); title(ax,'Secants approach the tangent as $h \\to 0$')
ax=axs[1]; ax.plot(tt,f(tt),color=BLUE,lw=2,label='$f(x)=x^2$'); ax.plot(tt,2*tt,color=RED,lw=1.8,ls='--',label="$f'(x)=2x$"); ax.plot(1,2,'o',color=RED,ms=5); ax.plot(1,1,'o',color=BLUE,ms=5)
ax.annotate("$f'(1)=2$",xy=(1,2),xytext=(1.5,0.6),fontsize=8.5,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.6)); ax.legend(fontsize=8.5,loc='upper left'); ax.set_xlim(-0.2,3.2); ax.set_ylim(-0.3,9.6)
ax.set_xlabel('$x$'); grid(ax); title(ax,'The slope is itself a function of $x$'); fig.tight_layout(w_pad=1.2); save(fig,'f311.png')

# ---- 3.2-1 gallery of ML functions and their derivatives
sg=lambda z:1/(1+np.exp(-z)); sp_=lambda z:np.log1p(np.exp(z))
fig,axs=plt.subplots(2,3,figsize=(4.31,2.72),dpi=240)
def panel(ax,xx,fv,dv,name,ylim,legend=False,xlim=None,lab=("$f$","$f'$")):
    ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5); ax.plot(xx,fv,color=BLUE,lw=1.9,label=lab[0]); ax.plot(xx,dv,color=RED,lw=1.6,ls='--',label=lab[1]); ax.set_ylim(*ylim); ax.set_xlim(xx[0],xx[-1]); title(ax,name); grid(ax)
z=np.linspace(-3,3,400)
panel(axs[0,0],z,z**2,2*z,'square: $z^2$,  slope $2z$',(-6.5,9.5)); axs[0,0].legend(fontsize=8,loc='lower right')
za=np.linspace(-3,3,401); panel(axs[0,1],za,np.abs(za),np.sign(za),'absolute value: $|z|$',(-1.6,4.2))
axs[0,1].plot([0,0],[-1,1],color=RED,lw=3,alpha=0.35,solid_capstyle='butt'); axs[0,1].plot([0],[1],'o',mfc='white',mec=RED,ms=4,zorder=5); axs[0,1].plot([0],[-1],'o',mfc='white',mec=RED,ms=4,zorder=5)
axs[0,1].text(0.2,4.0,'slopes in\n[$-1$, 1] at 0',fontsize=8,color=RED,va='top',ha='left')
panel(axs[0,2],z,np.exp(z),np.exp(z),'exponential: $e^z$,  slope $e^z$',(-2,12))
zp=np.linspace(0.12,4,300); panel(axs[1,0],zp,np.log(zp),1/zp,'logarithm: $\\ln z$,  slope $1/z$',(-2.6,5)); axs[1,0].set_xlim(0,4)
zs=np.linspace(-6,6,400); panel(axs[1,1],zs,sg(zs),sg(zs)*(1-sg(zs)),'sigmoid $\\sigma(z)$,  slope $\\sigma(1-\\sigma)$',(-0.15,1.15))
panel(axs[1,2],zs,sp_(zs),sg(zs),'softplus $\\ln(1+e^z)$,  slope $\\sigma(z)$',(-0.4,6.5))
for ax in axs[1]: ax.set_xlabel('$z$')
fig.tight_layout(h_pad=1.0,w_pad=0.8); save(fig,'f321.png')

# ---- 3.3-1 stationary points and the second derivative; delivery slice
fig,axs=plt.subplots(1,2,figsize=(4.05,1.95),dpi=240); ax=axs[0]; f=lambda w:w**3/3-w**2+1; ww=np.linspace(-0.9,3.3,300); ax.plot(ww,f(ww),color=BLUE,lw=2)
for w0,col,t1,t2 in [(0,RED,'maximum:','$f\'=0$, $f\'\'<0$'),(2,GREEN,'minimum:','$f\'=0$, $f\'\'>0$')]:
    ax.plot([w0-0.7,w0+0.7],[f(w0)]*2,color=col,lw=1.3); ax.plot(w0,f(w0),'o',color=col,ms=5,zorder=5)
ax.text(0.15,f(0)+0.3,'maximum\n$f\'=0$, $f\'\'<0$',fontsize=8.5,color=RED,va='bottom',ha='left'); ax.text(2.0,f(2)-0.3,'minimum\n$f\'=0$, $f\'\'>0$',fontsize=8.5,color=GREEN,va='top',ha='center')
ax.set_xlim(-0.9,3.3); ax.set_ylim(-2.2,3.0); ax.set_xlabel('$w$'); ax.set_ylabel('$f(w)$'); grid(ax); title(ax,'Flat slope: maximum or minimum?')
ax=axs[1]; wl=np.linspace(0.9,2.5,200); Jl=np.array([np.sum((y-0.9-w_*x)**2) for w_ in wl]); ax.plot(wl,Jl,color=RED,lw=2); ax.plot(1.7,1.1,'*',color=GREEN,ms=13,mec='white',zorder=5); ax.plot([1.7-0.25,1.7+0.25],[1.1,1.1],color=GREEN,lw=1.3)
ax.text(1.7,17,'$J\'(w)=110w-187$\n$J\'\'(w)=110>0$\nminimum at $w=1.7$',fontsize=8,color=GREEN,va='bottom',ha='center',linespacing=1.4); ax.set_xlabel('slope $w$  (with $b=0.9$)'); ax.set_ylabel('SSE  $J$'); ax.set_xlim(0.9,2.5); ax.set_ylim(-3,48); grid(ax); title(ax,'The delivery loss along $w$ (Example 3.3-1)')
fig.tight_layout(w_pad=1.2); save(fig,'f331.png')

# ---- 3.4-1 partial derivatives as slopes of slices
F=lambda u,v:(u-1)**2+3*(v-0.5)**2; P0=(2.0,1.0)
fig,axs=plt.subplots(1,3,figsize=(4.31,1.87),dpi=240,gridspec_kw={'width_ratios':[1.15,1,1]}); ax=axs[0]; uu=np.linspace(-0.6,3.2,200); vv=np.linspace(-0.6,1.8,200); Ug,Vg=np.meshgrid(uu,vv)
ax.contour(Ug,Vg,F(Ug,Vg),levels=[0.25,1,2,4,7],colors=BLUE,linewidths=0.9); ax.set_aspect('equal'); ax.plot([-0.6,3.2],[1,1],color=PUR,lw=1.6); ax.plot([2,2],[-0.6,1.8],color=PUR,lw=1.6,ls='--'); ax.plot(*P0,'o',color=RED,ms=5,zorder=5)
ax.text(2.05,1.05,'P',color=RED,fontsize=9,ha='left',va='bottom'); ax.set_xlabel('$u$'); ax.set_ylabel('$v$'); title(ax,'Contours of $f(u,v)$'); ax.set_xlim(-0.6,3.2); ax.set_ylim(-0.6,1.8)
ax=axs[1]; ax.plot(uu,F(uu,1.0),color=PUR,lw=1.8); ax.plot(uu,F(2,1)+2*(uu-2),color=GREEN,lw=1.2); ax.plot(2,F(2,1),'o',color=RED,ms=5); ax.set_ylim(0,8); ax.set_xlim(-0.6,3.2); ax.set_xlabel('$u$  ($v$ fixed at 1)'); ax.set_ylabel('$f$'); grid(ax); title(ax,'slope in $u$: $\\partial f/\\partial u=2$',)
ax=axs[2]; ax.plot(vv,F(2.0,vv),color=PUR,lw=1.8,ls='--'); ax.plot(vv,F(2,1)+3*(vv-1),color=GREEN,lw=1.2); ax.plot(1,F(2,1),'o',color=RED,ms=5); ax.set_ylim(0,8); ax.set_xlim(-0.6,1.8); ax.set_xlabel('$v$  ($u$ fixed at 2)'); grid(ax); title(ax,'slope in $v$: $\\partial f/\\partial v=3$')
fig.tight_layout(w_pad=0.8); save(fig,'f341.png'); NUM['f341']={'f_P':F(2,1),'fu':2*(2-1),'fv':6*(1-0.5)}

# ---- 3.5-1 gradient field of the delivery loss
bb=np.linspace(-3,4.8,400); ww=np.linspace(0,3.0,300); Bg,Wg=np.meshgrid(bb,ww); J=Jd(Bg,Wg)
fig,ax=plt.subplots(figsize=(4.05,1.95),dpi=240); cs=ax.contourf(Bg,Wg,J,levels=[1.1,3,8,20,50,120,300,800],cmap='Blues_r',alpha=0.9,norm=matplotlib.colors.LogNorm())
ax.contour(Bg,Wg,J,levels=[3,8,20,50,120,300],colors='white',linewidths=0.6); ax.set_aspect('equal')
gradJ=lambda b_,w_: np.array([-2*np.sum(y-b_-w_*x),-2*np.sum(x*(y-b_-w_*x))]); sc=0.0065
pts=[(0.0,1.0),(3.0,2.5),(-2.0,2.4),(4.0,1.0),(-1.2,0.65),(2.6,0.9)]
NUM['grad_field']={}
for p in pts:
    g_=gradJ(*p); NUM['grad_field'][str(p)]=[round(float(g_[0]),1),round(float(g_[1]),1)]
    u_=g_/np.linalg.norm(g_)*0.55; arr(ax,p,(p[0]+u_[0],p[1]+u_[1]),c=RED,lw=1.7,ms=9); ax.plot(*p,'o',color=NAVY,ms=3.5,zorder=6,mec='white',mew=0.5)
g0=gradJ(0.0,1.0); u0=g0/np.linalg.norm(g0)*0.55; arr(ax,(0,1),(-u0[0],1-u0[1]),c=GREEN,lw=1.7,ms=9)
ax.plot(0.9,1.7,'*',color=OR,ms=13,mec='white',zorder=7)
ax.text(0.45,0.95,'$J=51$ at this point',fontsize=8.5,color=NAVY,ha='left',va='center'); ax.text(1.05,1.68,'minimum',fontsize=8.5,color=NAVY,ha='left',va='bottom',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
ax.text(0.32,1.5,'$-\\nabla J$',fontsize=9,color='white',ha='left',va='center',bbox=dict(fc=GREEN,ec='none',pad=1.2)); ax.text(0.12,0.38,'$\\nabla J$',fontsize=9,color='white',ha='left',va='center',bbox=dict(fc=RED,ec='none',pad=1.2))
ax.set_xlim(-3,4.8); ax.set_ylim(0,3.0); ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); fig.tight_layout(); save(fig,'f351.png')

# ---- 3.6-1 computation graph, forward and backward
xv,yv,wv,bv=1.5,1.0,0.8,-0.5; u=wv*xv; z=u+bv; p=1/(1+np.exp(-z)); Lv=-(yv*np.log(p)+(1-yv)*np.log(1-p))
dLdp=-yv/p+(1-yv)/(1-p); dpdz=p*(1-p); dLdz=dLdp*dpdz; dLdw=dLdz*xv; dLdb=dLdz
NUM['graph']={k:round(float(v),4) for k,v in dict(u=u,z=z,p=p,L=Lv,dLdp=dLdp,dpdz=dpdz,dLdz=dLdz,dLdw=dLdw,dLdb=dLdb).items()}
mn=lambda v,d=3: f'{v:.{d}f}'.replace('-','\u2212')
fig,ax=plt.subplots(figsize=(4.31,1.78),dpi=240); ax.axis('off'); ax.set_xlim(-0.3,11.4); ax.set_ylim(-0.2,4.1)
nx=[2.9,5.3,7.7,10.0]; ny=2.0; names=['\u00d7','+','\u03c3','loss']
for xx_,nm in zip(nx,names): ax.add_patch(plt.Circle((xx_,ny),0.42,fc=LB,ec=BLUE,lw=1.1,zorder=3)); ax.text(xx_,ny,nm,ha='center',va='center',fontsize=10 if nm!='loss' else 7.5,color=NAVY,zorder=4)
def ib(xc,yc,t,c=BLUE): ax.add_patch(FancyBboxPatch((xc-0.65,yc-0.24),1.3,0.48,boxstyle='round,pad=0.01,rounding_size=0.08',fc=LGREY,ec=c,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=8.5,color=NAVY)
ib(0.7,3.45,f'$w$ = {wv}'); ib(0.7,ny,f'$x$ = {xv}'); ib(5.3,3.6,f'$b$ = {mn(bv,1)}'); ib(10.0,3.6,f'$y$ = {yv:.0f}')
A=lambda p_,q_,c=NAVY: arr(ax,p_,q_,c=c,lw=1.1,ms=8)
A((1.15,3.25),(2.55,ny+0.30)); A((1.15,ny),(2.48,ny)); A((5.15,3.36),(5.15,ny+0.43)); A((10.0,3.36),(10.0,ny+0.43))
for a_,b_,t in [(0,1,f'$u$ = {u:.2f}'),(1,2,f'$z$ = {z:.2f}'),(2,3,f'$p$ = {p:.4f}')]:
    A((nx[a_]+0.45,ny),(nx[b_]-0.45,ny)); ax.text((nx[a_]+nx[b_])/2,ny+0.2,t,ha='center',va='bottom',fontsize=8.5,color=BLUE)
ax.text(nx[3]+0.5,ny+0.0,f'$L$ = {Lv:.4f}',ha='left',va='center',fontsize=8.5,color=BLUE)
for a_,b_,t in [(3,2,f'$\\partial L/\\partial p$\n= {mn(dLdp)}'),(2,1,f'$\\partial L/\\partial z$\n= {mn(dLdz)}'),(1,0,f'$\\partial L/\\partial u$\n= {mn(dLdz)}')]:
    arr(ax,(nx[a_]-0.45,ny-0.3),(nx[b_]+0.45,ny-0.3),c=RED,lw=1.1,ms=8); ax.text((nx[a_]+nx[b_])/2,ny-0.42,t,ha='center',va='top',fontsize=8.5,color=RED)
arr(ax,(2.68,ny+0.46),(1.3,3.45),c=RED,lw=1.1,ms=8); ax.text(1.9,3.75,f'$\\partial L/\\partial w$ = {mn(dLdw)}',ha='left',va='center',fontsize=8.5,color=RED)
arr(ax,(5.5,ny+0.43),(5.5,3.35),c=RED,lw=1.1,ms=8); ax.text(5.7,3.0,f'$\\partial L/\\partial b$ = {mn(dLdb)}',ha='left',va='center',fontsize=8.5,color=RED)
ax.text(5.5,0.0,'blue: forward pass (values)          red: backward pass (derivatives of $L$)',ha='center',fontsize=8.5,color=GREY)
save(fig,'f361.png')

# ---- 3.7-1 gradients of a linear and a quadratic function
fig,axs=plt.subplots(1,2,figsize=(3.7,1.78),dpi=240); gg_=np.linspace(-2.2,2.2,200); G1,G2=np.meshgrid(gg_,gg_); a_=np.array([1.0,0.5]); Am=np.array([[2.0,1.0],[1.0,3.0]])
ax=axs[0]; ax.contour(G1,G2,a_[0]*G1+a_[1]*G2,levels=np.arange(-4,4.1,0.5),colors=BLUE,linewidths=0.9); ax.set_aspect('equal')
for p_ in [(-1.4,-1.0),(0.0,1.5),(1.5,-1.2),(-0.5,0.2),(1.2,0.6),(-1.5,1.4)]:
    arr(ax,p_,(p_[0]+0.7*a_[0],p_[1]+0.7*a_[1]),c=RED,lw=1.5,ms=8); ax.plot(*p_,'o',color=NAVY,ms=3)
ax.set_xlim(-2.2,2.2); ax.set_ylim(-2.2,2.2); ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$'); title(ax,'$\\mathbf{a}^{\\mathrm{T}}\\mathbf{w}$: the gradient is the same everywhere')
ax=axs[1]; Q=Am[0,0]*G1**2+2*Am[0,1]*G1*G2+Am[1,1]*G2**2; ax.contour(G1,G2,Q,levels=[0.5,1.5,3,5,8,12,17],colors=BLUE,linewidths=0.9); ax.set_aspect('equal')
for p_ in [(-1.5,0.4),(0.4,-1.2),(1.3,0.9),(-0.6,1.2),(0.6,0.2),(1.5,-0.5)]:
    gq=2*Am@np.array(p_); gq=0.13*gq; arr(ax,p_,(p_[0]+gq[0],p_[1]+gq[1]),c=RED,lw=1.5,ms=8); ax.plot(*p_,'o',color=NAVY,ms=3)
ax.set_xlim(-2.2,2.2); ax.set_ylim(-2.2,2.2); ax.set_xlabel('$w_1$'); title(ax,'$\\mathbf{w}^{\\mathrm{T}}\\mathbf{A}\\mathbf{w}$: the gradient is $2\\mathbf{A}\\mathbf{w}$')
fig.tight_layout(w_pad=1.0); save(fig,'f371.png')

# ---- 3.8-1 Hessian and curvature
H=2*Xd.T@Xd; lam,V=np.linalg.eigh(H); NUM['hess']={'H':H.tolist(),'lam':[round(float(l),2) for l in lam],'v_flat':[round(float(c),3) for c in V[:,0]],'v_steep':[round(float(c),3) for c in V[:,1]],'ratio':round(float(lam[1]/lam[0]),1)}
vf=V[:,0]*np.sign(V[0,0]); vs=V[:,1]*np.sign(V[1,1])
fig,axs=plt.subplots(1,2,figsize=(4.31,1.87),dpi=240,gridspec_kw={'width_ratios':[1.5,1]}); ax=axs[0]; b0,w0=0.9,1.7
bb=np.linspace(-3.2,5,400); ww=np.linspace(0.3,3.1,300); Bg,Wg=np.meshgrid(bb,ww); Jq=1.1+0.5*(H[0,0]*(Bg-b0)**2+2*H[0,1]*(Bg-b0)*(Wg-w0)+H[1,1]*(Wg-w0)**2)
ax.contour(Bg,Wg,Jq,levels=[2,4,8,16,32,64],colors=BLUE,linewidths=0.9); ax.set_aspect('equal'); ax.plot(b0,w0,'*',color=GREEN,ms=12,mec='white',zorder=6)
Lf=2.9; Ls=0.55
ax.plot([b0-Lf*vf[0],b0+Lf*vf[0]],[w0-Lf*vf[1],w0+Lf*vf[1]],color=PUR,lw=1.5); ax.plot([b0-Ls*vs[0],b0+Ls*vs[0]],[w0-Ls*vs[1],w0+Ls*vs[1]],color=OR,lw=1.5,ls='--')
ax.text(1.6,0.4,f'flat direction, $\\lambda$ = {lam[0]:.2f}',fontsize=8.5,color=PUR,ha='left',va='bottom',bbox=dict(fc='white',ec='none',pad=1,alpha=0.9)); ax.annotate(f'steep direction, $\\lambda$ = {lam[1]:.1f}',xy=(b0+0.5*vs[0],w0+0.5*vs[1]),xytext=(1.6,2.72),fontsize=8.5,color=OR,ha='left',va='center',bbox=dict(fc='white',ec='none',pad=1,alpha=0.9),arrowprops=dict(arrowstyle='-',color=OR,lw=0.7))
ax.set_xlim(-3.2,5); ax.set_ylim(0.3,3.1); ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); title(ax,'Contours of the delivery loss and the axes of $\\mathbf{H}$')
ax=axs[1]; s=np.linspace(-2.2,2.2,300); ax.plot(s,1.1+0.5*lam[0]*s**2,color=PUR,lw=1.8,label=f'flat: $\\lambda$ = {lam[0]:.2f}'); ax.plot(s,1.1+0.5*lam[1]*s**2,color=OR,lw=1.8,ls='--',label=f'steep: $\\lambda$ = {lam[1]:.1f}')
ax.set_ylim(0,14); ax.set_xlim(-2.2,2.2); ax.set_xlabel('distance $s$ from the minimum'); ax.set_ylabel('$J$'); ax.legend(fontsize=8,loc='upper right'); grid(ax); title(ax,'Cuts along the two axes')
fig.tight_layout(w_pad=1.0); save(fig,'f381.png')

# ---- 3.9-1 Taylor approximations of softplus at 0
sp_=lambda t:np.log1p(np.exp(t)); t1=lambda t:np.log(2)+0.5*t; t2=lambda t:np.log(2)+0.5*t+0.125*t**2
fig,axs=plt.subplots(1,2,figsize=(4.05,1.87),dpi=240,gridspec_kw={'width_ratios':[1.25,1]}); ax=axs[0]; tt=np.linspace(-3,3.2,300)
ax.plot(tt,sp_(tt),color=BLUE,lw=2.2,label='exact'); ax.plot(tt,t1(tt),color=PUR,lw=1.5,ls='--',label='first order (tangent)'); ax.plot(tt,t2(tt),color=GREEN,lw=1.5,label='second order (parabola)')
ax.plot(0,np.log(2),'o',color=RED,ms=5,zorder=5); ax.plot([1,1],[t1(1),t2(1)],color=RED,lw=0.9); ax.plot(1,sp_(1),'o',color=BLUE,ms=4,zorder=5)
ax.annotate('$x$ = 1',xy=(1,t1(1)),xytext=(1.5,0.55),fontsize=8.5,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.6))
ax.set_xlim(-3,3.2); ax.set_ylim(-0.2,4.2); ax.legend(fontsize=8,loc='upper left'); ax.set_xlabel('$x$'); ax.set_ylabel('softplus$(x)$'); grid(ax); title(ax,'Expansion about $x$ = 0')
ax=axs[1]; xe=np.linspace(0,3,200); ax.plot(xe,np.abs(sp_(xe)-t1(xe)),color=PUR,lw=1.6,ls='--',label='first order'); ax.plot(xe,np.abs(sp_(xe)-t2(xe)),color=GREEN,lw=1.6,label='second order')
ax.set_xlabel('distance $x$ from 0'); ax.set_ylabel('absolute error'); ax.legend(fontsize=8,loc='upper left'); grid(ax); title(ax,'The error grows with distance'); ax.set_xlim(0,3)
fig.tight_layout(w_pad=1.0); save(fig,'f391.png')

# ---- 3.10-1 step size and error of difference quotients
hs=10.0**-np.linspace(1,15,141); fwd=np.abs((np.exp(1+hs)-np.exp(1))/hs-np.e); cen=np.abs((np.exp(1+hs)-np.exp(1-hs))/(2*hs)-np.e)
i_best=int(np.argmin(cen)); NUM['best_h']={'h':float(hs[i_best]),'err':float(cen[i_best]),'fwd_best_h':float(hs[int(np.argmin(fwd))]),'fwd_best_err':float(fwd.min())}
fig,ax=plt.subplots(figsize=(2.99,1.95),dpi=240); ax.loglog(hs,fwd,color=PUR,lw=1.6,ls='--',label='forward difference'); ax.loglog(hs,cen,color=GREEN,lw=1.8,label='central difference')
ax.plot(hs[i_best],cen[i_best],'o',color=RED,ms=5,zorder=5); ax.invert_xaxis(); ax.set_xlabel('step $h$'); ax.set_ylabel('error of the slope of $e^x$ at 1'); ax.legend(fontsize=8,loc='upper center'); grid(ax)
ax.text(9e-2,2e-11,'truncation error',fontsize=8,color=GREY,ha='left',va='bottom'); ax.text(2e-9,3e-11,'round-off error',fontsize=8,color=GREY,ha='left',va='bottom')
ax.annotate('best $h$ about $8\\times10^{-6}$',xy=(hs[i_best],cen[i_best]),xytext=(1.5e-6,2e-13),va='bottom',ha='left',fontsize=8,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.6)); ax.set_ylim(1e-13,1)
fig.tight_layout(); save(fig,'f3101.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

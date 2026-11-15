# Problem figures of Chapter 3. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
A={}
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9,'legend.fontsize':9})
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10]); X=np.c_[np.ones(5),x]
sg=lambda z:1/(1+np.exp(-z)); sp=lambda z:np.log1p(np.exp(z))
nd=lambda f,t,h=1e-6:(f(t+h)-f(t-h))/(2*h)
def small(w=2.4,h=1.35): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
Jd=lambda v: float(np.sum((y-X@v)**2)); gJ=lambda v: -2*X.T@(y-X@v)
def delivery_contours(ax):
    bb=np.linspace(-3,4.5,300); ww=np.linspace(0,3,200); Bg,Wg=np.meshgrid(bb,ww)
    J=((y[None,None,:]-Bg[...,None]-Wg[...,None]*x[None,None,:])**2).sum(-1)
    ax.contour(Bg,Wg,J,levels=[2,8,20,50,120,300],colors=BLUE,linewidths=0.8); ax.set_aspect('equal'); ax.set_xlim(-3,4.5); ax.set_ylim(0,3)
    ax.set_xlabel('$b$',fontsize=9); ax.set_ylabel('$w$',fontsize=9); ax.tick_params(labelsize=9)
# 3A.1
f1=lambda t:(3*t**2+1)**4; f2=lambda t:t**2*np.exp(-t)
A['3A.1']={'a_exact':1536.0,'a_num':round(nd(f1,1.0),2),'b_exact':round(float(np.exp(-1)),4),'b_num':round(nd(f2,1.0),4)}
fig,axs=plt.subplots(1,2,figsize=(2.8,1.35),dpi=240); tt=np.linspace(0,2,200)
axs[0].plot(tt[:100],f1(tt[:100]),color=BLUE,lw=1.6); axs[0].set_title('$(3x^2+1)^4$',fontsize=9); axs[1].plot(tt,f2(tt),color=BLUE,lw=1.6); axs[1].set_title('$x^2e^{-x}$',fontsize=9)
for ax,f,x0 in zip(axs,[f1,f2],[1,1]): ax.plot(x0,f(x0),'o',color=RED,ms=4); ax.set_xlabel('$x$',fontsize=9); ax.tick_params(labelsize=9); gr(ax)
fig.tight_layout(w_pad=1); save(fig,'p_3A1.png')
# 3A.2
A['3A.2']={'s0':float(sg(0)),'slope0':0.25,'s2':round(float(sg(2)),4),'slope2':round(float(sg(2)*(1-sg(2))),4),'num_slope2':round(nd(sg,2.0),4)}
fig,ax=small(); zz=np.linspace(-6,6,300); ax.plot(zz,sg(zz),color=BLUE,lw=1.6,label='$\\sigma(z)$'); ax.plot(zz,sg(zz)*(1-sg(zz)),color=RED,lw=1.4,ls='--',label="$\\sigma'(z)$"); ax.legend(fontsize=9,loc='center left'); ax.set_xlabel('$z$',fontsize=9); gr(ax); ax.tick_params(labelsize=9); save(fig,'p_3A2.png')
# 3A.3
H3=np.array([[2.,1],[1,4]]); ev=np.linalg.eigvalsh(H3); A['3A.3']={'grad_at_1_m1':[1,-3],'hessian':H3.tolist(),'eig':[round(float(e),3) for e in ev]}
fig,ax=small(2.2,1.6); g_=np.linspace(-2,2,200); U,V=np.meshgrid(g_,g_); ax.contour(U,V,U**2+U*V+2*V**2,levels=[0.5,1.5,3,5,8],colors=BLUE,linewidths=0.8); ax.set_aspect('equal')
arr(ax,(1,-1),(1.3,-1.9),c=RED,lw=1.4,ms=8) if False else None
gg=np.array([1,-3.]); gg=0.5*gg/np.linalg.norm(gg)*1.4; arr(ax,(1,-1),(1+gg[0],-1+gg[1]),c=RED,lw=1.4,ms=8); ax.plot(1,-1,'o',color=NAVY,ms=3.5); ax.text(1.1,-0.85,'(1, $-1$)',fontsize=9,color=NAVY); ax.set_xlabel('$u$',fontsize=9); ax.set_ylabel('$v$',fontsize=9); ax.tick_params(labelsize=9); save(fig,'p_3A3.png')
# 3A.4
v=np.array([1.,2.]); gv=gJ(v); vn=v-0.01*gv; A['3A.4']={'J':Jd(v),'grad':gv.tolist(),'new':[round(float(c),3) for c in vn],'J_new':round(Jd(vn),3)}
fig,ax=small(2.6,1.35); delivery_contours(ax); ax.plot(*v,'o',color=NAVY,ms=3.5); ax.plot(*vn,'o',color=GREEN,ms=3.5); arr(ax,tuple(v),tuple(vn),c=GREEN,lw=1.2,ms=7); ax.text(v[0]+0.2,v[1]+0.25,'start (1, 2)',fontsize=9,color=NAVY); ax.plot(0.9,1.7,'*',color=OR,ms=8); save(fig,'p_3A4.png')
# 3B.1
al=2.0; gr_an=gJ(np.array([0.,1.]))+2*al*np.array([0.,1.]); rid=lambda t:Jd(t)+al*float(t@t)
vv=np.array([0.,1.]); gr_num=np.array([nd(lambda s,k=k:rid(vv+s*np.eye(2)[k]),0.0) for k in range(2)])
# note: penalty here includes the intercept as well, so analytic = grad J + 2*alpha*v
gr_an=gJ(vv)+2*al*vv; A['3B.1']={'analytic':gr_an.tolist(),'numeric':[round(float(c),3) for c in gr_num]}
fig,ax=small(2.4,1.35); lab=['$\\partial/\\partial b$','$\\partial/\\partial w$']; idx=np.arange(2); ax.bar(idx-0.18,gr_an,0.34,color=BLUE,label='analytic'); ax.bar(idx+0.18,gr_num,0.34,color=OR,label='numeric',hatch='//',edgecolor='white')
ax.set_xticks(idx); ax.set_xticklabels(lab,fontsize=9); ax.axhline(0,color=GREY,lw=0.6); ax.legend(fontsize=9,loc='lower left'); ax.tick_params(labelsize=9); gr(ax); save(fig,'p_3B1.png')
# 3B.2
x0=1.0; f0,d1,d2=float(sp(x0)),float(sg(x0)),float(sg(x0)*(1-sg(x0))); xq=1.5; t2=f0+d1*(xq-x0)+0.5*d2*(xq-x0)**2
A['3B.2']={'f':round(f0,4),'f1':round(d1,4),'f2':round(d2,4),'exact_1.5':round(float(sp(xq)),4),'second_order':round(t2,4),'error':round(float(sp(xq))-t2,5)}
fig,ax=small(); tt=np.linspace(-1,3,200); ax.plot(tt,sp(tt),color=BLUE,lw=1.6); ax.plot(tt,f0+d1*(tt-x0)+0.5*d2*(tt-x0)**2,color=GREEN,lw=1.3); ax.plot(x0,f0,'o',color=RED,ms=4); ax.set_xlabel('$x$',fontsize=9); ax.tick_params(labelsize=9); ax.set_ylim(0,4); gr(ax); save(fig,'p_3B2.png')
# 3B.3
hs=10.0**-np.arange(1,15); err=np.abs((np.sin(1+hs)-np.sin(1-hs))/(2*hs)-np.cos(1)); kb=int(np.argmin(err)); A['3B.3']={'best_k':int(kb+1),'best_err':float(err[kb]),'err_h1e-3':float(err[2]),'err_h1e-12':float(err[11])}
fig,ax=small(); ax.loglog(hs,err,'-o',color=GREEN,ms=3,lw=1.2); ax.invert_xaxis(); ax.plot(hs[kb],err[kb],'o',color=RED,ms=5); ax.set_xlabel('$h$',fontsize=9); ax.set_ylabel('error',fontsize=9); ax.tick_params(labelsize=9); gr(ax); save(fig,'p_3B3.png')
# 3C.1 abs kink
fig,ax=small(); zz=np.linspace(-1.5,1.5,200); ax.plot(zz,np.abs(zz),color=BLUE,lw=1.6)
for s_ in [-0.8,-0.3,0.4,0.9]: ax.plot([-0.9,0.9],[s_*(-0.9),s_*0.9],color=GREEN,lw=0.8,alpha=0.8)
ax.plot(0,0,'o',color=RED,ms=4); ax.set_ylim(-0.3,1.6); ax.set_xlabel('$z$',fontsize=9); ax.tick_params(labelsize=9); gr(ax); ax.text(0.55,-0.15,'lines with slopes in [$-1$, 1]\nstay below the curve',fontsize=9,color=GREEN,va='top',ha='center') if False else None; save(fig,'p_3C1.png')
# 3C.2 saturation
fig,ax=small(); zz=np.linspace(-10,10,300); ax.semilogy(zz,sg(zz)*(1-sg(zz)),color=RED,lw=1.5); ax.set_xlabel('$z$',fontsize=9); ax.set_ylabel("$\\sigma'(z)$",fontsize=9); ax.tick_params(labelsize=9); gr(ax); save(fig,'p_3C2.png')
A['3C.2']={'slope_at_10':float(sg(10)*(1-sg(10))),'slope_at_0':0.25}
# 3C.3 gradient does not point to the minimum
fig,ax=small(2.6,1.35); delivery_contours(ax); p0=np.array([3.0,1.0]); gq=gJ(p0); u=-gq/np.linalg.norm(gq)*1.1; arr(ax,tuple(p0),tuple(p0+u),c=GREEN,lw=1.3,ms=7); ax.plot(*p0,'o',color=NAVY,ms=3.5); ax.plot(0.9,1.7,'*',color=OR,ms=9); ax.annotate('minimum',xy=(0.9,1.7),xytext=(-2.9,2.55),fontsize=9,color=OR,arrowprops=dict(arrowstyle='-',color=OR,lw=0.6)); ax.text(3.25,1.5,'$-\\nabla J$',fontsize=9,color=GREEN,ha='left',va='center'); save(fig,'p_3C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

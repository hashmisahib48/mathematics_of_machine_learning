# Problem figures of Chapter 5. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from scipy.optimize import minimize
A={}
x=np.arange(1.,6); y=np.array([3.,4,6,7,10]); X=np.c_[np.ones(5),x]; H=2*X.T@X
J=lambda v:float(((y-X@v)**2).sum()); grad=lambda v:-2*X.T@(y-X@v)
def small(w=2.3,h=1.3): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 5A.1
f=lambda w:3*(w-2)**2+1; P=[0.]
for _ in range(3): P.append(P[-1]-0.1*6*(P[-1]-2))
A['5A.1']={'w':[round(v,4) for v in P[1:]],'factor':0.4,'limit':round(1/3,4)}
fig,ax=small(); w=np.linspace(-0.4,3.6,200); ax.plot(w,f(w),color=BLUE,lw=1.6); ax.plot(P,[f(v) for v in P],'o-',color=NAVY,ms=4,lw=0.8,zorder=5)
for k,v in enumerate(P[:2]): ax.text(v,f(v)+1.8,f'$w_{k}$',fontsize=9,color=NAVY,ha='center')
ax.set_xlabel('$w$'); ax.set_ylabel('$f(w)$'); ax.set_ylim(0,17); gr(ax); save(fig,'p_5A1.png')
# 5A.2
v1=np.zeros(2)-0.005*grad(np.zeros(2)); v2=v1-0.005*grad(v1)
A['5A.2']={'w1':v1.round(4).tolist(),'J0':J(np.zeros(2)),'J1':round(J(v1),4),'w2':v2.round(4).tolist(),'J2':round(J(v2),4),'g1':grad(v1).round(4).tolist()}
fig,ax=small(2.3,1.0); bb=np.linspace(-0.5,3,200); ww=np.linspace(-0.3,3,200); Bg,Wg=np.meshgrid(bb,ww)
Z=((y[None,None,:]-Bg[...,None]-Wg[...,None]*x[None,None,:])**2).sum(-1); ax.contour(Bg,Wg,Z,levels=[1.1,2,4,8,16,32,64,128,256],colors=BLUE,linewidths=0.6)
Pp=np.array([[0,0],v1,v2]); ax.plot(Pp[:,0],Pp[:,1],'o-',color=NAVY,ms=4,lw=0.9,zorder=5); ax.plot(0.9,1.7,'*',color=GREEN,ms=11,mec='white')
ax.set_xlabel('$b$'); ax.set_ylabel('$w$'); save(fig,'p_5A2.png')
# 5A.3
fe=lambda w:np.exp(w)-2*w; P=[0.]
for _ in range(3): P.append(P[-1]-(np.exp(P[-1])-2)/np.exp(P[-1]))
A['5A.3']={'w':[round(v,6) for v in P[1:]],'wstar':round(float(np.log(2)),6)}
fig,ax=small(); w=np.linspace(-0.6,1.6,200); ax.plot(w,fe(w),color=BLUE,lw=1.6); ax.plot(P[:3],[fe(v) for v in P[:3]],'o',color=NAVY,ms=4,zorder=5)
for k in range(2): ax.text(P[k],fe(P[k])+0.35,f'$w_{k}$',fontsize=9,color=NAVY,ha='center')
ax.plot(np.log(2),fe(np.log(2)),'*',color=GREEN,ms=11,mec='white'); ax.set_xlabel('$w$'); ax.set_ylabel('$f(w)$'); ax.set_ylim(0,3.6); gr(ax); save(fig,'p_5A3.png')
# 5A.4
A['5A.4']={'min':[-1,-2],'min_value':-5,'max':[1,2],'max_value':5,'lambda_min':0.5}
fig,ax=small(2.3,1.55); th=np.linspace(0,2*np.pi,300); r5=np.sqrt(5); ax.plot(r5*np.cos(th),r5*np.sin(th),color=RED,lw=1.6)
xx=np.array([-3,3]); 
for c_ in (-5,0,5): ax.plot(xx,(c_-xx)/2,color=BLUE,lw=0.9,ls='-' if abs(c_)==5 else ':')
ax.plot([-1,1],[-2,2],'o',color=GREEN,ms=5,zorder=5); ax.text(-1.0,-2.55,'min',fontsize=9,color=GREEN,ha='center'); ax.text(1.0,2.25,'max',fontsize=9,color=GREEN,ha='center'); ax.set_aspect('equal'); ax.set_xlim(-3,3); ax.set_ylim(-3,3)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); gr(ax); save(fig,'p_5A4.png')
# 5B.1 momentum
def heavy(eta,mu,n):
    v=np.zeros(2);u=np.zeros(2);L=[J(v)]
    for _ in range(n): u=mu*u-eta*grad(v); v=v+u; L.append(J(v))
    return np.array(L)
lam=np.linalg.eigvalsh(H); Lp=heavy(1/lam[-1],0.0,300); Lm=heavy(0.008,0.8,300)
first=lambda L:int(np.where(L<1.1+0.01)[0][0])
A['5B.1']={'plain':first(Lp),'momentum':first(Lm),'eta_plain':round(float(1/lam[-1]),5)}
fig,ax=small(2.3,1.3); ax.semilogy(Lp-1.1+1e-6,color=NAVY,lw=1.4,label='plain'); ax.semilogy(Lm-1.1+1e-6,color=OR,lw=1.4,ls='--',label='momentum'); ax.set_xlim(0,150); ax.set_ylim(1e-4,300)
ax.set_xlabel('step $t$'); ax.set_ylabel('$J-J_{\\min}$'); ax.legend(fontsize=9); gr(ax); save(fig,'p_5B1.png')
# 5B.2 scaling n_iter
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
Xc,yc=make_classification(n_samples=300,n_features=5,random_state=0); Xb=Xc.copy(); Xb[:,0]*=1000
def fit(D):
    with warnings.catch_warnings(record=True) as cw:
        warnings.simplefilter('always'); m=LogisticRegression().fit(D,yc)
    return int(m.n_iter_[0]),any(w.category is ConvergenceWarning for w in cw)
r1=fit(Xc); r2=fit(Xb); r3=fit(StandardScaler().fit_transform(Xb))
A['5B.2']={'plain':r1,'scaled_by_1000':r2,'standardised':r3}
fig,ax=small(2.3,1.3); ax.bar(['unit\nscale','x1000','x1000,\nstandardised'],[r1[0],r2[0],r3[0]],color=[BLUE,OR,GREEN],width=0.6); ax.axhline(100,color=RED,lw=1.0,ls='--'); ax.text(2.4,103,'max_iter',fontsize=9,color=RED,ha='right'); ax.set_ylabel('n_iter_'); ax.set_ylim(0,125); ax.tick_params(axis='x',labelsize=9); save(fig,'p_5B2.png')
# 5B.3 Nelder-Mead vs BFGS
fJ=lambda v:J(v)
nm=minimize(fJ,[0,0],method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-10}); bf=minimize(fJ,[0,0],jac=grad,method='BFGS')
A['5B.3']={'nelder_mead_nfev':int(nm.nfev),'nelder_mead_x':nm.x.round(4).tolist(),'bfgs_nfev':int(bf.nfev),'bfgs_njev':int(bf.njev),'bfgs_x':bf.x.round(4).tolist()}
fig,ax=small(2.3,1.3); ax.bar(['Nelder-\nMead','BFGS'],[nm.nfev,bf.nfev+bf.njev],color=[OR,BLUE],width=0.5); ax.set_ylabel('evaluations'); ax.tick_params(axis='x',labelsize=9); save(fig,'p_5B3.png')
# 5C figures
fig,ax=small(2.3,1.3); z=np.linspace(-3,3,300); ax.plot(z,np.sin(2*z)+0.15*z**2,color=BLUE,lw=1.5); ax.plot(2.189,np.sin(4.378)+0.15*2.189**2,'o',color=OR,ms=5); ax.text(0.3,-0.34,'saddle or\nlocal minimum?',fontsize=9,color=OR,va='top',linespacing=1.0); ax.set_xlabel('$w$'); ax.set_ylabel('$J$'); gr(ax); save(fig,'p_5C1.png')
fig,ax=small(2.3,1.3); c=np.array([3,4.]); 
for r_,c_ in ((1,RED),(2.5,PUR)): ax.plot(r_*np.cos(th),r_*np.sin(th),color=c_,lw=1.5)
for r_ in (2,3.2,4.2): ax.plot(3+r_*np.cos(th),4+r_*np.sin(th),color=BLUE,lw=0.6)
ax.plot(*c,'o',color=BLUE,ms=4); ax.text(0,0,'$t$',color=RED,fontsize=9,ha='center'); ax.set_aspect('equal'); ax.set_xlim(-3,7.6); ax.set_ylim(-3,8.4); ax.axis('off'); save(fig,'p_5C2.png')
rng=np.random.default_rng(2); t=np.arange(200); fig,ax=small(2.3,1.3); ax.plot(t,0.2+rng.normal(0,0.12,200).cumsum()*0+0.2*np.abs(rng.normal(size=200)),color=RED,lw=0.8); ax.plot(t,0.2*np.exp(-t/25)+0.02,color=GREEN,lw=1.6); ax.set_xlabel('update $t$'); ax.set_ylabel('loss'); gr(ax); save(fig,'p_5C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

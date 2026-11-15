# Problem figures of Chapter 21. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,sys; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel, sigmoid_kernel, polynomial_kernel
from krr_scratch import MyKRR
A={}
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
def small(w=2.5,h=1.8): return plt.subplots(figsize=(w,h),dpi=240)
def grd(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 21A.1 six products of the feature map
def phi(v): a,b,s=v[0],v[1],np.sqrt(2); return np.array([1,s*a,s*b,a*a,s*a*b,b*b])
x=np.array([2.,1]); z=np.array([1.,3]); pr=phi(x)*phi(z)
A['21A.1']={'phi_x':[round(float(v),4) for v in phi(x)],'phi_z':[round(float(v),4) for v in phi(z)],'products':[round(float(v),4) for v in pr],'sum':round(float(pr.sum()),4),'poly':float((x@z+1)**2)}
fig,ax=small(2.5,1.7); ax.bar(range(6),pr,color=BLUE,width=0.6)
for i,v in enumerate(pr): ax.text(i,v+0.25,f'{v:.0f}',ha='center',fontsize=9)
ax.set_xticks(range(6)); ax.set_xticklabels([f'$\\varphi_{i+1}$' for i in range(6)]); ax.set_ylim(0,15); ax.set_ylabel('product'); ax.spines[['top','right']].set_visible(False); grd(ax); save(fig,'p_21A1.png')
# 21A.4 Laplacian Gram matrix and a quadratic form
Xl=np.array([[0.],[1.],[2.]]); Kl=laplacian_kernel(Xl,gamma=1.0); c=np.array([1.,-1,1]); q=float(c@Kl@c)
A['21A.4']={'K':np.round(Kl,4).tolist(),'quad':round(q,4),'eigs':[round(float(v),4) for v in np.linalg.eigvalsh(Kl)]}
fig,ax=small(2.2,1.9); ax.imshow(Kl,cmap='Blues',vmin=0,vmax=1.2)
for i in range(3):
    for j in range(3): ax.text(j,i,f'{Kl[i,j]:.2f}',ha='center',va='center',fontsize=9.5,color='white' if Kl[i,j]>0.6 else NAVY)
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['0','1','2']); ax.set_yticklabels(['0','1','2']); ax.spines[['top','right']].set_visible(True); ax.set_title('$e^{-|x-z|}$ at 0, 1, 2',fontsize=9.5,color=NAVY); save(fig,'p_21A4.png')
# 21A.2 two points, linear kernel
x2=np.array([1.,2]); y2=np.array([2.,3]); K2=np.outer(x2,x2); a2=np.linalg.solve(K2+np.eye(2),y2); w2=float(a2@x2)
A['21A.2']={'a':[round(float(v),4) for v in a2],'w':round(w2,4),'w_primal':round(float((x2@y2)/(x2@x2+1)),4),'f3':round(3*w2,4)}
fig,ax=small(2.4,1.8); tt=np.array([0,3.4]); ax.plot(tt,w2*tt,color=GREEN,lw=1.8); ax.scatter(x2,y2,s=36,color=BLUE,edgecolor='white',zorder=5); ax.scatter([3],[3*w2],s=36,marker='D',color='white',edgecolor=PUR,zorder=6)
ax.text(1.55,4.35,'$f(3)$ = ?',color=PUR,fontsize=9,ha='center'); ax.set_xlim(0,3.4); ax.set_ylim(0,5); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); grd(ax); save(fig,'p_21A2.png')
# 21A.3 one training point
al,g=1.0,1.0; f1=lambda t: 2/(1+al)*np.exp(-g*t**2)
A['21A.3']={'f0':float(f1(0)),'f1':round(float(f1(1)),4),'f2':round(float(f1(2)),4),'check_sklearn':round(float(KernelRidge(alpha=1,kernel='rbf',gamma=1).fit([[0.]],[2.]).predict([[1.]])[0]),4)}
fig,ax=small(2.5,1.8); tt=np.linspace(-3,3,200); ax.plot(tt,f1(tt),color=GREEN,lw=1.8); ax.scatter([0],[2],s=36,color=BLUE,edgecolor='white',zorder=5); ax.plot([0,0],[1,2],color=RED,lw=1.2)
ax.text(0.2,1.75,'shrunk by $\\alpha$',color=RED,fontsize=9); ax.set_xlabel('$x$'); ax.set_ylabel('$f(x)$'); ax.set_ylim(0,2.4); grd(ax); save(fig,'p_21A3.png')
# 21B.1 sigmoid kernel in MyKRR
class Sig(MyKRR):
    def _k(self,A_,B_):
        if self.kernel=='sigmoid': return np.tanh(self.gamma*(A_@B_.T)+self.coef0)
        return super()._k(A_,B_)
r=np.random.default_rng(0); Xs=r.normal(size=(40,3)); ys=np.sin(Xs[:,0])+Xs[:,1]**2; Xn=r.normal(size=(5,3))
import warnings
kw=dict(alpha=1.0,kernel='sigmoid',gamma=0.5,coef0=0.0); o=Sig(**kw).fit(Xs,ys)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter('always'); rf=KernelRidge(**kw).fit(Xs,ys)
warn_txt=[str(w_.message) for w_ in wl]
Kg=sigmoid_kernel(Xs,gamma=0.5,coef0=0.0)
A['21B.1']={'max_diff':float(abs(o.predict(Xn)-rf.predict(Xn)).max()),'min_eig':round(float(np.linalg.eigvalsh(Kg).min()),4),'n_negative':int((np.linalg.eigvalsh(Kg)<-1e-10).sum()),'warning':warn_txt,'min_eig_plus_alpha':round(float(np.linalg.eigvalsh(Kg+np.eye(40)).min()),4)}
fig,ax=small(2.15,1.55); zz=np.linspace(-4,4,200)
for gg,c_ in [(0.5,GREEN),(2,PUR)]: ax.plot(zz,np.tanh(gg*1*zz),color=c_,lw=1.6,label=f'$\\gamma$ = {gg:g}')
ax.legend(loc='upper left',handlelength=1.3); ax.set_xlabel('$z$'); ax.set_ylabel('$k(1, z)$'); ax.set_ylim(-1.2,1.2); grd(ax); save(fig,'p_21B1.png')
# 21B.2 leave-one-out by the hat matrix
rng=np.random.default_rng(4); xs=np.sort(rng.uniform(0,6,28)); yy=np.sin(xs)+0.25*rng.normal(size=28); g,al=0.1,0.03
K=rbf_kernel(xs[:,None],gamma=g); H=K@np.linalg.inv(K+al*np.eye(28)); loo_h=(yy-H@yy)/(1-np.diag(H)); loo=np.zeros(28)
for i in range(28):
    m=np.ones(28,bool); m[i]=False; f=KernelRidge(alpha=al,kernel='rbf',gamma=g).fit(xs[m,None],yy[m]); loo[i]=yy[i]-f.predict(xs[i:i+1,None])[0]
A['21B.2']={'max_diff':float(abs(loo-loo_h).max()),'loo_rmse':round(float(np.sqrt(np.mean(loo**2))),4),'train_rmse':round(float(np.sqrt(np.mean((yy-H@yy)**2))),4),'df':round(float(np.trace(H)),3)}
fig,ax=small(2.15,1.55); ax.scatter(xs,yy-H@yy,s=14,color=RED,label='ordinary residual'); ax.scatter(xs,loo,s=14,facecolor='none',edgecolor=OR,label='leave-one-out',marker='s'); ax.axhline(0,color=GREY,lw=0.7)
ax.legend(loc='lower left',handletextpad=0.1,fontsize=8.5,borderaxespad=0.1); ax.set_ylim(-1.1,0.6); ax.set_xlabel('$x$'); ax.set_ylabel('residual'); grd(ax); save(fig,'p_21B2.png')
# 21B.3 sum of kernels
rng=np.random.default_rng(3); xt=np.sort(rng.uniform(0,6,30)); yt=np.sin(xt)+0.5*xt+0.1*rng.normal(size=30); xe=np.linspace(0,8,200); truth=np.sin(xe)+0.5*xe
def fitpred(Kfun):
    Ktr=Kfun(xt[:,None],xt[:,None]); a=np.linalg.solve(Ktr+0.1*np.eye(30),yt); return Kfun(xe[:,None],xt[:,None])@a
lin=lambda P,Q: P@Q.T
preds={'rbf':fitpred(lambda P,Q: rbf_kernel(P,Q,gamma=1.0)),'linear':fitpred(lin),'rbf + linear':fitpred(lambda P,Q: rbf_kernel(P,Q,gamma=1.0)+P@Q.T)}
ext=xe>6; A['21B.3']={k:{'rmse_inside':round(float(np.sqrt(np.mean((v-truth)[~ext]**2))),3),'rmse_extrap':round(float(np.sqrt(np.mean((v-truth)[ext]**2))),3)} for k,v in preds.items()}
fig,ax=small(3.6,1.5); ax.plot(xe,truth,color=GREY,lw=0.9,ls=':'); ax.plot(xe,preds['rbf'],color=PUR,lw=1.4,label='RBF'); ax.plot(xe,preds['rbf + linear'],color=GREEN,lw=1.6,label='RBF + linear'); ax.scatter(xt,yt,s=8,color=BLUE,zorder=4); ax.axvline(6,color=GRID,lw=0.8)
ax.legend(loc='upper left',handlelength=1.3,borderaxespad=0.1); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); grd(ax); save(fig,'p_21B3.png')
# 21C.2 spectrum of the Gram matrix
fig,ax=small(2.15,1.55); 
for gg,c_ in [(0.05,PUR),(0.5,GREEN),(5,OR)]:
    ev=np.sort(np.linalg.eigvalsh(rbf_kernel(xs[:,None],gamma=gg)))[::-1]; ax.semilogy(range(1,29),np.maximum(ev,1e-16),color=c_,lw=1.4,label=f'$\\gamma$ = {gg:g}')
ax.axhline(0.1,color=RED,lw=0.9,ls='--'); ax.text(28,0.15,'$\\alpha$ = 0.1',color=RED,fontsize=9,ha='right'); ax.set_ylim(1e-16,1e2); ax.set_xlabel('index $j$'); ax.set_ylabel('eigenvalue $\\lambda_j$'); ax.legend(loc='lower left',fontsize=8.5,handlelength=1.3,borderaxespad=0.1,frameon=True,framealpha=0.92,edgecolor='none',facecolor='white'); grd(ax); save(fig,'p_21C2.png')
A['21C.2']={str(gg):[float(f'{v:.3g}') for v in np.sort(np.linalg.eigvalsh(rbf_kernel(xs[:,None],gamma=gg)))[::-1][[0,4,9,19]]] for gg in (0.05,0.5,5)}
# 21C.1 sizes of the two systems
fig,ax=new(2.5,1.9); ax.add_patch(Rectangle((0.0,0.0),0.35,0.35,fc=LB,ec=BLUE,lw=1)); ax.add_patch(Rectangle((0.7,0.0),1.9,1.9,fc='#E3F4EA',ec=GREEN,lw=1))
ax.text(0.175,-0.12,'$p\\times p$',ha='center',va='top',fontsize=9.5,color=BLUE); ax.text(1.65,-0.12,'$n\\times n$',ha='center',va='top',fontsize=9.5,color=GREEN); ax.text(0.175,0.45,'primal\n$\\Phi^{\\mathrm{T}}\\Phi+\\alpha I$',ha='center',va='bottom',fontsize=8.8,color=BLUE); ax.text(1.65,0.95,'dual\n$K+\\alpha I$',ha='center',va='center',fontsize=9.5,color=GREEN)
ax.set_xlim(-0.2,2.7); ax.set_ylim(-0.4,2.1); save(fig,'p_21C1.png')
# 21C.3 extrapolation
rng=np.random.default_rng(4); xs2=np.sort(rng.uniform(0,6,28)); ys2=np.sin(xs2)+0.25*rng.normal(size=28); gx=np.linspace(-3,9,300)
fig,ax=small(2.5,1.8); m=ys2.mean()
fr=KernelRidge(alpha=0.1,kernel='rbf',gamma=0.5).fit(xs2[:,None],ys2-m); fp=KernelRidge(alpha=0.1,kernel='poly',degree=3,gamma=0.1,coef0=1).fit(xs2[:,None],ys2-m)
ax.plot(gx,np.clip(fr.predict(gx[:,None])+m,-3,3),color=GREEN,lw=1.5,label='RBF'); ax.plot(gx,np.clip(fp.predict(gx[:,None])+m,-3,3),color=PUR,lw=1.3,ls='--',label='cubic'); ax.scatter(xs2,ys2,s=8,color=BLUE,zorder=4)
ax.set_ylim(-3,3); ax.legend(loc='upper left',handlelength=1.3,borderaxespad=0.1); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); grd(ax); save(fig,'p_21C3.png')
A['21C.3']={'rbf_at_-3':round(float(fr.predict([[-3.]])[0]+m),3),'rbf_at_9':round(float(fr.predict([[9.]])[0]+m),3),'mean':round(float(m),3),'cubic_at_9':round(float(fp.predict([[9.]])[0]+m),2)}
with open('verify/problem_answers.json','w') as fh: json.dump(A,fh,indent=1)
print('ok')

# 21C.1 primal or dual: memory and operation counts
import json as _j
_n,_p,_m=50000,20,1771
_pd={'K_GB':_n*_n*8/1e9,'K_ops':_n**3/3,'K_build':_n*_n*_p,'Phi_GB':_n*_m*8/1e9,'gram_ops':_n*_m**2,'solve_ops':_m**3/3,'XtX_MB':_m*_m*8/1e6}
print({k:'%.3g'%v for k,v in _pd.items()})
with open('verify/primal_dual.json','w') as _f: _j.dump(_pd,_f)

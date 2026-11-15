# Problem figures of Chapter 11. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error')
from figstyle import *
from sklearn.linear_model import (Lasso, ElasticNet, LinearRegression, MultiTaskLasso, OrthogonalMatchingPursuit, lasso_path, LassoCV)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
A={}
dist=np.array([1.,2,3,4,5]); lights=np.array([2.,1,4,3,5]); t=np.array([3.,4,6,7,10]); X=np.c_[dist,lights]; Xc=X-X.mean(0); yc=t-t.mean(); n=5
def small(w=2.4,h=1.7): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def soft(z,a): return np.sign(z)*max(abs(z)-a,0.0)

# 11A.1 one feature, alpha_max and alpha=2.5
c1=float(Xc[:,0]@yc/n); v1=float((Xc[:,0]**2).mean()); a=2.5; w=soft(c1,a)/v1; b=t.mean()-w*dist.mean()
m=Lasso(alpha=a,tol=1e-12).fit(dist[:,None],t)
A['11A.1']={'alpha_max':c1,'w':round(w,4),'b':round(b,4),'sk_w':round(float(m.coef_[0]),4),'sk_b':round(float(m.intercept_),4),'pred6':round(b+6*w,3)}
fig,ax=small(1.8,1.33); xx=np.array([0,6]); ax.plot(xx,0.9+1.7*xx,color=BLUE,lw=1.2,ls='--',label='least squares'); ax.plot(xx,b+w*xx,color=GREEN,lw=1.8,label='lasso, $\\alpha=2.5$')
ax.scatter(dist,t,s=22,color=BLUE,edgecolor='white',lw=0.7,zorder=5); gr(ax); ax.set_xlabel('distance (km)'); ax.set_ylabel('time (min)'); ax.legend(fontsize=8.5,loc='upper left',borderpad=0.2,handlelength=1.4,labelspacing=0.2); ax.set_xlim(0,6); ax.set_ylim(0,19); ax.set_yticks([0,5,10,15])
fig.tight_layout(pad=0.4); save(fig,'p_11A1.png')

# 11A.2 CD staircase at alpha = 2 on the toy (two features)
def J(w1,w2,al,rho=1.0):
    r=yc-Xc[:,0]*w1-Xc[:,1]*w2; return (r**2).sum()/(2*n)+al*rho*(abs(w1)+abs(w2))+0.5*al*(1-rho)*(w1**2+w2**2)
al=2.0; w=np.zeros(2); r=yc.copy(); v=(Xc**2).mean(0); steps=[tuple(w)]; zs=[]
for sweep in range(2):
    for j in range(2):
        z=Xc[:,j]@r/n+v[j]*w[j]; wn=soft(z,al)/v[j]; zs.append(round(float(z),4)); r+=Xc[:,j]*(w[j]-wn); w[j]=wn; steps.append(tuple(w))
mm=Lasso(alpha=al,tol=1e-12).fit(X,t); A['11A.2']={'z':zs,'w':[round(float(q),4) for q in w],'sk':[round(float(q),4) for q in mm.coef_],'b':round(float(t.mean()-w@X.mean(0)),4),'corr2':round(float(abs(Xc[:,1]@(yc-Xc@w))/n),4)}
g1,g2=np.meshgrid(np.linspace(-0.4,1.4,200),np.linspace(-0.6,1.0,200)); JJ=np.vectorize(lambda a_,b_:J(a_,b_,al))(g1,g2)
fig,ax=small(1.8,1.35); ax.contour(g1,g2,JJ,levels=JJ.min()+np.array([0.15,0.4,0.8,1.5,2.5,4]),colors=RED,linewidths=0.7,alpha=0.8)
st=np.array(steps); ax.plot(st[:,0],st[:,1],'-o',color=GREEN,ms=3.5,lw=1.4); ax.plot(0,0,'o',color=NAVY,ms=4); ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
ax.set_xlabel('$w_1$'); ax.set_ylabel('$w_2$'); ax.set_xlim(-0.4,1.4); ax.set_ylim(-0.6,1.0); fig.tight_layout(pad=0.4); save(fig,'p_11A2.png')

# 11A.3 elastic net, one feature, alpha=1, rho=0.5
rho=0.5; al=1.0; wen=soft(c1,al*rho)/(v1+al*(1-rho)); ben=t.mean()-wen*dist.mean(); me=ElasticNet(alpha=al,l1_ratio=rho,tol=1e-12).fit(dist[:,None],t)
A['11A.3']={'w':round(wen,4),'b':round(ben,4),'sk_w':round(float(me.coef_[0]),4),'sk_b':round(float(me.intercept_),4),'lasso_w':round(soft(c1,al)/v1,4)}
fig,ax=small(); ww=np.linspace(-0.2,2.2,300); f=lambda w_,rho_: 0.5*v1*w_**2-c1*w_+al*rho_*np.abs(w_)+0.5*al*(1-rho_)*w_**2
ax.plot(ww,f(ww,1.0),color=GREEN,lw=1.5,label='lasso'); ax.plot(ww,f(ww,0.5),color=OR,lw=1.5,ls='--',label='$\\rho=0.5$')
for rho_,w_,c_,mk in [(1.0,soft(c1,1)/v1,GREEN,'o'),(0.5,wen,OR,'s')]: ax.plot(w_,f(w_,rho_),mk,color=c_,ms=5,mec='white',zorder=6)
gr(ax); ax.set_xlabel('$w$'); ax.set_ylabel('$J(w)$ + constant'); ax.legend(fontsize=8.5,loc='upper right',borderpad=0.2); ax.set_ylim(-3.2,2); fig.tight_layout(pad=0.4); save(fig,'p_11A3.png')

# 11A.4 group threshold
z=np.array([6.0,8.0]); v=2.0; al=5.0; wg=max(1-al/np.linalg.norm(z),0)*z/v
xx=np.array([-1.,-1,1,1,-1,-1,1,1])  # build data with v=2? use direct formula check via MultiTaskLasso on x with v=2
x=np.array([-1.,-1,1,1])*np.sqrt(2); Yg=np.c_[x*3.0*np.sqrt(2)/np.sqrt(2)*1.0,x*4.0]  # c = x'Y/n = (z1,z2)/... check below
# direct: want x'x/n = 2 -> x=sqrt2*(-1,-1,1,1);  z_k = x'y_k/n -> y_k = z_k/sqrt2 * (-1,-1,1,1)... set y_k = z_k/np.sqrt(2)*sgn
sgn=np.array([-1.,-1,1,1]); Yg=np.c_[6.0/np.sqrt(2)*sgn,8.0/np.sqrt(2)*sgn]
mt=MultiTaskLasso(alpha=al,tol=1e-12).fit(x[:,None],Yg)
A['11A.4']={'norm_z':float(np.linalg.norm(z)),'factor':max(1-al/np.linalg.norm(z),0),'w':[round(float(q),4) for q in wg],'sk':[round(float(q),4) for q in mt.coef_.ravel()],'v_check':float((x**2).mean()),'z_check':[round(float(x@Yg[:,k]/4),4) for k in (0,1)],'separate':[round(soft(q,al)/v,4) for q in z],'alpha_zero':float(np.linalg.norm(z))}
fig,ax=small(2.8,1.75); th=np.linspace(0,2*np.pi,200); ax.plot(al*np.cos(th),al*np.sin(th),color=PUR,lw=1.2); ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
arr(ax,(0,0),tuple(z),c=BLUE,lw=1.4,ms=9); arr(ax,(0,0),tuple(z*(1-al/np.linalg.norm(z))),c=GREEN,lw=1.6,ms=9)
ax.text(5.6,8.2,'$\\mathbf{z}$',color=BLUE,fontsize=10); ax.text(0.2,3.4,'$v\\mathbf{w}$',color=GREEN,fontsize=10,ha='right'); ax.text(-6.0,6.2,'radius $\\alpha$',color=PUR,fontsize=9)
ax.set_aspect('equal'); ax.set_xlim(-6,9); ax.set_ylim(-6,9.5); ax.axis('off'); fig.tight_layout(pad=0.2); save(fig,'p_11A4.png')

# 11B.1 scaling: distance in metres
Xm=np.c_[1000*dist,lights]; lm=Lasso(alpha=1.0,tol=1e-12,max_iter=100000).fit(Xm,t); lk=Lasso(alpha=1.0,tol=1e-12).fit(X,t)
Xs=StandardScaler().fit_transform(Xm); ls_=Lasso(alpha=1.0,tol=1e-12).fit(Xs,t); Xs2=StandardScaler().fit_transform(X); ls2=Lasso(alpha=1.0,tol=1e-12).fit(Xs2,t)
A['11B.1']={'km':[round(float(q),4) for q in lk.coef_],'m':[round(float(q),6) for q in lm.coef_],'m_pred_equal':bool(np.allclose(lm.predict(Xm),lk.predict(X))),'std_m':[round(float(q),4) for q in ls_.coef_],'std_km':[round(float(q),4) for q in ls2.coef_],
  'pred_m':[round(float(q),3) for q in lm.predict(Xm)],'pred_km':[round(float(q),3) for q in lk.predict(X)]}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.7),dpi=240,sharey=True)
for ax,cf,tt in zip(axs,[lk.coef_,lm.coef_*1000],['km','m  (weight $\\times 1000$)']):
    ax.bar([0,1],cf,color=[BLUE,OR],width=0.6); ax.set_xticks([0,1]); ax.set_xticklabels(['$w_1$','$w_2$'],fontsize=9); ax.set_title(tt,fontsize=8.5,color=NAVY); gr(ax)
axs[0].set_ylabel('weight'); fig.tight_layout(pad=0.4,w_pad=0.4); save(fig,'p_11B1.png')

# 11B.3 path by warm-started CD vs lasso_path
sys.path.insert(0,'code'); 
def soft_v(z,a): return np.sign(z)*np.maximum(np.abs(z)-a,0.0)
def my_path(X_,y_,alphas,sweeps=2000,tol=1e-13):
    Xc_=X_-X_.mean(0); yc_=y_-y_.mean(); n_=len(y_); v_=(Xc_**2).mean(0); w_=np.zeros(X_.shape[1]); out=[]
    for al_ in alphas:                       # warm start: w_ carries over
        r_=yc_-Xc_@w_
        for s_ in range(sweeps):
            big=0.0
            for j in range(len(w_)):
                z_=Xc_[:,j]@r_/n_+v_[j]*w_[j]; wn=soft_v(z_,al_)/v_[j]; r_+=Xc_[:,j]*(w_[j]-wn); big=max(big,abs(wn-w_[j])); w_[j]=wn
            if big<tol: break
        out.append(w_.copy())
    return np.array(out).T
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0)
al_grid=np.geomspace(2.4,0.01,30); mine=my_path(Xtr,ytr,al_grid); _,ref,_=lasso_path(Xtr-Xtr.mean(0),ytr-ytr.mean(),alphas=al_grid,tol=1e-12,max_iter=100000)
A['11B.3']={'max_diff':float(np.abs(mine-ref).max()),'n_alphas':30}
fig,ax=small(2.6,1.8); 
for j in range(10): ax.plot(al_grid,ref[j],color=GREEN,lw=1.3,alpha=0.9); ax.plot(al_grid,mine[j],color=OR,lw=0,marker='.',ms=3)
ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xlabel('$\\alpha$'); ax.set_ylabel('weights'); gr(ax); ax.set_title('line: lasso_path;  dots: your code',fontsize=8.5,color=GREY); fig.tight_layout(pad=0.4); save(fig,'p_11B3.png')

# 11B.2 OMP scale sensitivity
omp1=OrthogonalMatchingPursuit(n_nonzero_coefs=1).fit(X,t); X10=np.c_[dist,10*lights]; omp2=OrthogonalMatchingPursuit(n_nonzero_coefs=1).fit(X10,t)
Xs3=StandardScaler().fit_transform(X10); omp3=OrthogonalMatchingPursuit(n_nonzero_coefs=1).fit(Xs3,t)
sc1=np.abs(Xc.T@yc); sc2=np.abs((X10-X10.mean(0)).T@yc)
A['11B.2']={'raw_coef':[round(float(q),4) for q in omp1.coef_],'scaled10_coef':[round(float(q),4) for q in omp2.coef_],'std_coef':[round(float(q),4) for q in omp3.coef_],'scores_raw':[float(q) for q in sc1],'scores_x10':[float(q) for q in sc2],
  'std_scores':[round(float(q),3) for q in np.abs(Xs3.T@(t-t.mean()))]}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.7),dpi=240,sharey=True)
for ax,sc,tt in zip(axs,[sc1,sc2],['lights as counted','lights $\\times$ 10']):
    ax.bar([0,1],sc,color=[BLUE,OR],width=0.6); ax.set_xticks([0,1]); ax.set_xticklabels(['$x_1$','$x_2$'],fontsize=9); ax.set_title(tt,fontsize=8.5,color=NAVY); gr(ax)
axs[0].set_ylabel('score $|\\mathbf{x}_j^{\\mathrm{T}}\\mathbf{r}|$'); fig.tight_layout(pad=0.4,w_pad=0.4); save(fig,'p_11B2.png')

# 11C.2 identical columns
Xd=np.c_[dist,dist]; l2=Lasso(alpha=1.0,tol=1e-12).fit(Xd,t); e2=ElasticNet(alpha=1.0,l1_ratio=0.5,tol=1e-12).fit(Xd,t)
l2b=Lasso(alpha=1.0,tol=1e-12,selection='random',random_state=3).fit(Xd,t); l2c=Lasso(alpha=1.0,tol=1e-12,positive=True).fit(Xd,t)
A['11C.2']={'lasso':[round(float(q),4) for q in l2.coef_],'lasso_sum':round(float(l2.coef_.sum()),4),'enet':[round(float(q),4) for q in e2.coef_],'enet_sum':round(float(e2.coef_.sum()),4),'lasso_random':[round(float(q),4) for q in l2b.coef_]}
fig,ax=small(2.4,1.7); ax.bar([0,1],l2.coef_,width=0.28,color=GREEN,label='lasso'); ax.bar([0.32,1.32],e2.coef_,width=0.28,color=OR,label='elastic net'); ax.set_xticks([0.16,1.16]); ax.set_xticklabels(['$w_1$','$w_2$'],fontsize=9)
gr(ax); ax.legend(fontsize=8.5,loc='upper right',borderpad=0.2); ax.set_ylabel('weight'); ax.set_ylim(0,1.5); fig.tight_layout(pad=0.4); save(fig,'p_11C2.png')

# 11C.1 selection frequency under bootstrap
rng=np.random.default_rng(0); B=60; freq=np.zeros(10); acv=0.0787
for bb in range(B):
    idx=rng.integers(0,len(ytr),len(ytr)); mdl=Lasso(alpha=acv,max_iter=100000,tol=1e-7).fit(Xtr[idx],ytr[idx]); freq+=(mdl.coef_!=0)
freq/=B; A['11C.1']={'freq':{nm:round(float(q),2) for nm,q in zip(d.feature_names,freq)},'B':B,'alpha':acv}
fig,ax=small(2.6,1.8); ax.bar(range(10),freq,color=[MAG if nm in('s1','s2') else GREEN for nm in d.feature_names],width=0.7); ax.set_xticks(range(10)); ax.set_xticklabels(d.feature_names,fontsize=8,rotation=90)
ax.set_ylabel('fraction of fits'); ax.set_ylim(0,1.12); gr(ax); fig.tight_layout(pad=0.4); save(fig,'p_11C1.png')

# 11C.3 alpha_max as a function of rho
Xc_d=Xtr-Xtr.mean(0); yc_d=ytr-ytr.mean(); amax1=float(np.abs(Xc_d.T@yc_d).max()/len(ytr)); rhos=np.array([1,0.5,0.2,0.1,0.05,0.01])
A['11C.3']={'alpha_max_rho1':round(amax1,4),'alpha_max':{str(r_):round(amax1/r_,2) for r_ in rhos}}
try: ElasticNet(l1_ratio=0.0)  # constructing is fine; the grid is the issue
except Exception as e: pass
from sklearn.linear_model import ElasticNetCV
try: ElasticNetCV(l1_ratio=0.0,cv=3).fit(Xtr,ytr); A['11C.3']['error']='none'
except Exception as e: A['11C.3']['error']=type(e).__name__+': '+str(e)[:110]
fig,ax=small(2.6,1.8); rr=np.linspace(0.04,1,200); ax.plot(rr,amax1/rr,color=PUR,lw=1.6); ax.plot(rhos,amax1/rhos,'o',color=PUR,ms=3.5); ax.set_xlabel('l1_ratio $\\rho$'); ax.set_ylabel('$\\alpha_{\\max}$'); gr(ax); ax.set_ylim(0,62)
fig.tight_layout(pad=0.4); save(fig,'p_11C3.png')
with open('verify/problem_answers.json','w') as fh: json.dump(A,fh,indent=1)
print(json.dumps(A,indent=1))

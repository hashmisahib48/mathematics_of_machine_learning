# Problem figures of Chapter 10. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
A={}
def small(w=2.4,h=1.65): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
xa=np.array([0.,1,2,3]); ya=np.array([1.,3,4,6]); xc=xa-1.5; yc=ya-3.5
# 10A.1
ws={a:float(xc@yc/(xc@xc+a)) for a in (0,5,15)}; A['10A.1']={'w':{str(a):round(v,4) for a,v in ws.items()},'b':{str(a):round(3.5-v*1.5,4) for a,v in ws.items()},'alpha_for_w1':float(xc@yc/1.0-xc@xc)}
fig,ax=small(); t=np.array([-0.3,3.3])
for a,c,ls in [(0,BLUE,'-'),(5,GREEN,'--'),(15,PUR,':')]: ax.plot(t,3.5-ws[a]*1.5+ws[a]*t,color=c,ls=ls,lw=1.6,label=f'$\\alpha$ = {a}')
ax.scatter(xa,ya,s=30,color=BLUE,edgecolor='white',zorder=5); gr(ax); ax.legend(fontsize=9,loc='upper left'); ax.set_xlabel('$x$',fontsize=10); ax.set_ylabel('$y$',fontsize=10); ax.set_ylim(0,7.5); ax.set_xlim(-0.3,3.3); save(fig,'p_10A1.png')
# 10A.2
G=np.array([[5.,4],[4,5]]); g=np.array([9.,3.]); ev=np.linalg.eigvalsh(G)
A['10A.2']={'w0':list(np.round(np.linalg.solve(G,g),4)),'w1':list(np.round(np.linalg.solve(G+np.eye(2),g),4)),'eig':list(ev),'f':list(np.round(ev/(ev+1),4)),'df':round(float((ev/(ev+1)).sum()),4)}
fig,ax=small(2.4,1.8); gx,gy=np.meshgrid(np.linspace(-1,5,200),np.linspace(-4,2,200)); w0=np.linalg.solve(G,g); dx=gx-w0[0]; dy=gy-w0[1]
ax.contour(gx,gy,G[0,0]*dx*dx+2*G[0,1]*dx*dy+G[1,1]*dy*dy,levels=[1,4,9],colors=BLUE,linewidths=0.9)
ax.plot(*w0,'o',color=BLUE,ms=5); ax.plot(*np.linalg.solve(G+np.eye(2),g),'*',color=GREEN,ms=10); ax.plot(0,0,'+',color=NAVY); ax.set_xlabel('$w_1$',fontsize=10); ax.set_ylabel('$w_2$',fontsize=10); gr(ax); ax.set_aspect('equal'); ax.set_xlim(-1,5); ax.set_ylim(-4,2); save(fig,'p_10A2.png')
# 10A.3
wo=np.array([3,-2,0.5]); A['10A.3']={'a1':list(wo/2),'a3':list(wo/4),'df1':1.5,'df3':0.75}
fig,ax=small(2.4,1.65); xx=np.arange(3); ax.bar(xx-0.27,wo,0.27,color=BLUE,label='$\\alpha$ = 0'); ax.bar(xx,wo/2,0.27,color=GREEN,label='$\\alpha$ = 1'); ax.bar(xx+0.27,wo/4,0.27,color=PUR,label='$\\alpha$ = 3')
ax.axhline(0,color=GREY,lw=0.6); ax.set_xticks(xx); ax.set_xticklabels(['$w_1$','$w_2$','$w_3$'],fontsize=10); ax.legend(fontsize=8.5,loc='upper right'); gr(ax); save(fig,'p_10A3.png')
# 10A.4
a=5; w=xc@yc/(xc@xc+a); r=yc-w*xc; h=0.25+xc**2/(xc@xc+a); e=r/(1-h); ex=[]
for i in range(4):
    m=np.arange(4)!=i; ex.append(ya[i]-Ridge(alpha=a).fit(xa[m,None],ya[m]).predict(xa[i:i+1,None])[0])
A['10A.4']={'r':list(np.round(r,4)),'h':list(np.round(h,4)),'e':list(np.round(e,4)),'loo':round(float((e**2).mean()),4),'explicit_ok':bool(np.allclose(e,ex))}
fig,ax=small(); ax.scatter(xa,ya,s=30,color=BLUE,edgecolor='white',zorder=5); ax.plot(t,3.5-w*1.5+w*t,color=GREEN,lw=1.6)
m=np.arange(4)!=0; mm=Ridge(alpha=a).fit(xa[m,None],ya[m]); ax.plot(t,mm.predict(t[:,None]),color=OR,lw=1.3,ls='--'); ax.plot(0,ya[0],'o',mfc='white',mec=RED,ms=9,mew=1.4,zorder=6)
ax.text(1.0,0.6,'dashed: fit without\nthe first point',fontsize=9,color=OR); gr(ax); ax.set_xlabel('$x$',fontsize=10); ax.set_ylabel('$y$',fontsize=10); ax.set_ylim(0,7.5); ax.set_xlim(-0.3,3.3); save(fig,'p_10A4.png')
# 10B.1
d=load_diabetes(); X,y=d.data,d.target; alphas=np.logspace(-3,1,20)
Xc=X-X.mean(0); yc2=y-y.mean(); U,s,Vt=np.linalg.svd(Xc,full_matrices=False); z=U.T@yc2
W=np.array([Vt.T@(s/(s**2+a)*z) for a in alphas]); R=np.array([Ridge(alpha=a).fit(X,y).coef_ for a in alphas]); A['10B.1']={'max_diff':float(np.abs(W-R).max())}
fig,ax=small()
for j,nm,c,ls in [(2,'bmi',BLUE,'-'),(4,'s1',OR,'--'),(5,'s2',GREEN,'-.')]: ax.plot(alphas,W[:,j],lw=1.5,color=c,ls=ls,label=nm)
ax.set_xscale('log'); gr(ax); ax.set_xlabel('$\\alpha$',fontsize=10); ax.set_ylabel('$w_j$',fontsize=10); ax.legend(fontsize=8.5,loc='center left'); save(fig,'p_10B1.png')
# 10B.2
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=0); rng=np.random.default_rng(0); dup=X[:,2]+0.01*X[:,2].std()*rng.normal(size=len(X)); X2=np.c_[X,dup]
X2tr,X2te=train_test_split(X2,test_size=0.25,random_state=0)[:2]
mo=LinearRegression().fit(X2tr,ytr); mr=Ridge(alpha=0.05).fit(X2tr,ytr)
A['10B.2']={'ols_bmi':[round(float(mo.coef_[2]),1),round(float(mo.coef_[10]),1)],'ridge_bmi':[round(float(mr.coef_[2]),1),round(float(mr.coef_[10]),1)],'ols_test':round(float(mo.score(X2te,yte)),3),'ridge_test':round(float(mr.score(X2te,yte)),3),'corr':round(float(np.corrcoef(X[:,2],dup)[0,1]),5)}
fig,ax=small(); ax.scatter(X[:,2],dup,s=5,color=BLUE,alpha=0.6,lw=0); ax.plot([-0.1,0.2],[-0.1,0.2],color=GREEN,lw=1); ax.set_xlabel('bmi',fontsize=10); ax.set_ylabel('bmi + noise',fontsize=10); gr(ax); save(fig,'p_10B2.png')
# 10B.3
Xs=X.copy(); Xs[:,2]*=100; Xstr,Xste=train_test_split(Xs,test_size=0.25,random_state=0)[:2]
m1=Ridge(alpha=0.05).fit(Xtr,ytr); m2=Ridge(alpha=0.05).fit(Xstr,ytr); m3=make_pipeline(StandardScaler(),Ridge(alpha=0.05)).fit(Xstr,ytr); m4=make_pipeline(StandardScaler(),Ridge(alpha=0.05)).fit(Xtr,ytr)
A['10B.3']={'orig_test':round(float(m1.score(Xte,yte)),3),'bmi100_test':round(float(m2.score(Xste,yte)),3),'bmi100_w_bmi':round(float(m2.coef_[2]),2),'orig_w_bmi':round(float(m1.coef_[2]),1),'scaled_test':round(float(m3.score(Xste,yte)),3),'scaled_orig_test':round(float(m4.score(Xte,yte)),3)}
fig,ax=small(); ax.bar([0,1],[A['10B.3']['orig_test'],A['10B.3']['bmi100_test']],width=0.5,color=[BLUE,OR]); ax.set_xticks([0,1]); ax.set_xticklabels(['original','bmi $\\times$ 100'],fontsize=9); ax.set_ylabel('test $R^2$',fontsize=10); gr(ax); save(fig,'p_10B3.png')
# 10C.1 intercept penalised?
fig,ax=small(2.4,1.65); xx=np.array([1,2,3,4,5.]); yy=np.array([3,4,6,7,10.]); t2=np.array([0,5.5]); ax.scatter(xx,yy+100,s=28,color=BLUE,edgecolor='white',zorder=5); ax.scatter(xx,yy,s=28,color=GREY,edgecolor='white',zorder=5)
ax.plot(t2,0.9+1.7*t2,color=GREY,lw=1.2); ax.plot(t2,100.9+1.7*t2,color=GREEN,lw=1.2); ax.text(0.2,45,'targets shifted\nby 100',fontsize=9,color=BLUE); ax.set_xlabel('$x$',fontsize=10); ax.set_ylabel('$y$',fontsize=10); gr(ax); save(fig,'p_10C1.png')
# 10C.2 LOO curve ends at smallest alpha
fig,ax=small(2.4,1.65); aa=np.logspace(-1,1,30); xx1=xx[:,None]
def loo(a):
    Xc_=xx1-xx1.mean(0); S=np.linalg.inv(Xc_.T@Xc_+a*np.eye(1)); h=0.2+np.einsum('ij,jk,ik->i',Xc_,S,Xc_); r=(yy-yy.mean())-Xc_@(S@Xc_.T@(yy-yy.mean())); return np.mean((r/(1-h))**2)
ax.plot(aa,[loo(a) for a in aa],color=RED,lw=1.5); ax.set_xscale('log'); ax.set_xlabel('$\\alpha$',fontsize=10); ax.set_ylabel('LOO error',fontsize=10); gr(ax); save(fig,'p_10C2.png')
# 10C.3 p > n
Xw,yw=make_regression(n_samples=60,n_features=100,n_informative=100,noise=20,random_state=0); Xa,ya2,Xb,yb=Xw[:30],yw[:30],Xw[30:],yw[30:]
al=np.logspace(-3,3,25); trn=[Ridge(alpha=a).fit(Xa,ya2).score(Xa,ya2) for a in al]; tst=[Ridge(alpha=a).fit(Xa,ya2).score(Xb,yb) for a in al]
A['10C.3']={'best_alpha':round(float(al[int(np.argmax(tst))]),3),'best_test':round(float(max(tst)),3),'train_small_alpha':round(float(trn[0]),3),'test_small_alpha':round(float(tst[0]),3)}
fig,ax=small(2.4,1.65); ax.plot(al,trn,color=BLUE,lw=1.4,label='training'); ax.plot(al,tst,color=GREY,lw=1.4,ls=':',label='test'); ax.set_xscale('log'); ax.legend(fontsize=8.5,loc='center left'); ax.set_xlabel('$\\alpha$   ($n$ = 30, $p$ = 100)',fontsize=9); ax.set_ylabel('$R^2$',fontsize=10); gr(ax); save(fig,'p_10C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

# Problem figures and answers of Chapter 13. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.filterwarnings('ignore')
from figstyle import *
from sklearn.linear_model import *
from sklearn.datasets import make_regression
from huber_scratch import MyHuber
A={}
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,20.]); X1=x[:,None]
def small(w=2.4,h=1.7): return plt.subplots(figsize=(w,h),dpi=240)
def sc(ax,xx,yy,s=24): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
F=9.2
# 13A.1 Huber weights
sg,eps=2.0,1.35; rr=np.array([-1.,2,3,-6,12]); om=np.minimum(1,eps*sg/np.abs(rr)); A['13A.1']={'eps_sigma':eps*sg,'omega':[round(float(v),4) for v in om]}
fig,ax=small(); zz=np.linspace(0.05,8,300); ax.plot(zz,np.minimum(1,eps/zz),color=GREEN,lw=1.6); z_=np.abs(rr)/sg; ax.plot(z_,om,'o',color=BLUE,ms=5,mec='white'); ax.set_xlabel('$|r|/\\sigma$',fontsize=F); ax.set_ylabel('weight $\\omega$',fontsize=F); ax.grid(color=GRID,lw=0.5); ax.set_ylim(0,1.12); ax.tick_params(labelsize=F-0.5); save(fig,'p_13A1.png')
# 13A.2 trials
f_,s_,g_=0.6,3,0.99; N1=int(np.ceil(np.log(1-g_)/np.log(1-f_**s_))); N2=int(np.ceil(np.log(1-0.999)/np.log(1-f_**s_))); N3=int(np.ceil(np.log(1-g_)/np.log(1-f_**2)))
A['13A.2']={'f_s':f_**s_,'N_099':N1,'N_0999':N2,'N_s2':N3,'raw':float(np.log(1-g_)/np.log(1-f_**s_))}
fig,ax=small(); ff=np.linspace(0.3,0.95,100)
for s2,c,ls in ((2,BLUE,'-'),(3,GREEN,'--'),(5,PUR,':')): ax.plot(ff,np.ceil(np.log(0.01)/np.log(1-ff**s2)),color=c,ls=ls,lw=1.5,label=f'$s$ = {s2}')
ax.set_yscale('log'); ax.legend(fontsize=F-0.7,loc='upper right',borderpad=0.1,handlelength=1.8); ax.set_xlabel('inlier share $f$',fontsize=F); ax.set_ylabel('trials $N$',fontsize=F); ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=F-0.5); save(fig,'p_13A2.png')
# 13A.3 Theil-Sen by hand
xs=np.arange(5.); ys=np.array([1.,3,5,7,30]); sl=[(ys[j]-ys[i])/(xs[j]-xs[i]) for i in range(5) for j in range(i+1,5)]; w_=float(np.median(sl)); b_=float(np.median(ys-w_*xs)); mo=LinearRegression().fit(xs[:,None],ys)
A['13A.3']={'slopes_sorted':[round(v,3) for v in sorted(sl)],'w':w_,'b':b_,'ols_w':round(float(mo.coef_[0]),2),'ols_b':round(float(mo.intercept_),2)}
fig,ax=small(); sc(ax,xs,ys); xx=np.array([-0.2,4.2]); ax.plot(xx,b_+w_*xx,color=GREEN,lw=1.6); ax.plot(xx,mo.intercept_+mo.coef_[0]*xx,color=GREY,lw=1.4,ls='--'); ax.set_xlabel('$x$',fontsize=F); ax.set_ylabel('$y$',fontsize=F); ax.tick_params(labelsize=F-0.5); ax.set_ylim(-3,33); save(fig,'p_13A3.png')
# 13A.4 pinball
d=np.array([1.,2,3,4,10]); tau=0.25
def pj(c): r_=d-c; return float(np.where(r_>=0,tau*r_,(tau-1)*r_).sum())
A['13A.4']={'c2':pj(2),'c3':pj(3),'table':{str(c):pj(c) for c in (1,2,3,4,10)}}
fig,ax=small(); cc=np.linspace(0,11,200); ax.plot(cc,[pj(c) for c in cc],color=PUR,lw=1.6); ax.plot([2],[pj(2)],'o',color=GREEN,ms=6,mec='white',zorder=5); ax.plot(d,np.full(5,-0.6),'|',color=BLUE,ms=7,mew=1.4); ax.set_ylim(-1.5,None); ax.set_xlabel('constant $c$',fontsize=F); ax.set_ylabel('$\\sum\\rho_{0.25}$',fontsize=F); ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=F-0.5); save(fig,'p_13A4.png')
# 13B.1 MyHuber eps sweep
A['13B.1']={}
for e in (1.2,1.35,1.4,1.5):
    m=MyHuber(epsilon=e,alpha=0.0,n_iter=2000).fit(X1,y); r=HuberRegressor(epsilon=e,alpha=0,max_iter=2000,tol=1e-10).fit(X1,y); A['13B.1'][str(e)]={'mine':[round(float(m.coef_[0]),3),round(float(m.intercept_),3)],'sk':[round(float(r.coef_[0]),3),round(float(r.intercept_),3)]}
fig,ax=new(2.6,1.6); box(ax,0,0.9,1.1,0.5,'toy data',fc=LGREY,ec=GREY,fs=F); box(ax,1.5,0.9,1.5,0.5,'MyHuber($\\epsilon$)',fc=LB,ec=BLUE,fs=F); box(ax,1.5,0.1,1.5,0.5,'slope $w$',fc='#E3F4EA',ec=GREEN,fs=F); arr(ax,(1.1,1.15),(1.5,1.15),c=NAVY); arr(ax,(2.25,0.9),(2.25,0.6),c=NAVY); ax.set_xlim(-0.1,3.1); ax.set_ylim(0,1.5); save(fig,'p_13B1.png')
# 13B.2 quantile alpha
A['13B.2']={}
for a in (0,0.01,0.1,0.5,1.0):
    q=QuantileRegressor(alpha=a).fit(X1,y); A['13B.2'][str(a)]=[round(float(q.coef_[0]),3),round(float(q.intercept_),3)]
fig,ax=small(); sc(ax,x,y); xx=np.array([0,5.4])
for a,c,ls in ((0,GREEN,'-'),(0.1,PUR,'--'),(1.0,GREY,':')): q=QuantileRegressor(alpha=a).fit(X1,y); ax.plot(xx,q.intercept_+q.coef_[0]*xx,color=c,ls=ls,lw=1.5)
ax.text(0.1,16,'$\\alpha$ = 0, 0.1, 1',fontsize=F-0.5,color=PUR); ax.set_xlabel('$x$',fontsize=F); ax.set_ylabel('$y$',fontsize=F); ax.tick_params(labelsize=F-0.5); save(fig,'p_13B2.png')
# 13C.1 slope vs epsilon on toy
es=np.linspace(1.15,1.7,56); sl_=[HuberRegressor(epsilon=e,alpha=0,max_iter=3000,tol=1e-10).fit(X1,y).coef_[0] for e in es]
fig,ax=small(); ax.plot(es,sl_,color=GREEN,lw=1.6); ax.axhline(3.7,color=GREY,lw=1,ls='--'); ax.text(1.16,3.35,'least squares',fontsize=F-0.7,color=GREY,va='top'); ax.set_xlabel('$\\epsilon$',fontsize=F); ax.set_ylabel('Huber slope',fontsize=F); ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=F-0.5); ax.set_ylim(1.4,4.1); save(fig,'p_13C1.png')
A['13C.1']={'jump_between':[float(es[np.argmax(np.diff(sl_))]),float(es[np.argmax(np.diff(sl_))+1])]}
# 13C.2 coverage with small sample
rng=np.random.default_rng(11)
def mk(n): x_=rng.uniform(0,10,n); return x_[:,None],2+1.5*x_+rng.normal(size=n)*(0.3+0.35*x_)
Xs,ys_=mk(30); Xt,yt=mk(4000); cov=[]
fig,ax=small(); ax.scatter(Xs[:,0],ys_,s=10,color=BLUE,lw=0); xg=np.linspace(0,10,20)[:,None]; A['13C.2']={}
for t_ in (0.05,0.95):
    q=QuantileRegressor(quantile=t_,alpha=0).fit(Xs,ys_); ax.plot(xg[:,0],q.predict(xg),color=PUR,lw=1.5,ls='--'); A['13C.2'][str(t_)]=round(float(np.mean(yt<=q.predict(Xt))),3)
A['13C.2']['band']=round(A['13C.2']['0.95']-A['13C.2']['0.05'],3)
ax.set_xlabel('$x$',fontsize=F); ax.set_ylabel('$y$',fontsize=F); ax.tick_params(labelsize=F-0.5); ax.grid(color=GRID,lw=0.5); save(fig,'p_13C2.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

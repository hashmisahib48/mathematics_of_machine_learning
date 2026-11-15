# Problem figures of Chapter 6. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from numpy.polynomial import Polynomial
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
A={}
def small(w=2.1,h=1.5): return plt.subplots(figsize=(w,h),dpi=240)
def fin(ax,xl='',yl='',s=8.5):
    ax.set_xlabel(xl,fontsize=9); ax.set_ylabel(yl,fontsize=9); ax.tick_params(labelsize=s,length=2.5); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# ---- 6A.1 train and test errors
xt=np.array([0.,1,2,3]); yt=np.array([1.,2,2,4]); xs=np.array([4.,5,6]); ys=np.array([6.,5,8]); f=lambda t:1+t
jt=float(np.mean((yt-f(xt))**2)); js=float(np.mean((ys-f(xs))**2)); A['6A.1']={'train':jt,'test':js,'gap':js-jt}
fig,ax=small(); tt=np.array([-0.3,6.4]); ax.plot(tt,f(tt),color=GREEN,lw=1.6); ax.scatter(xt,yt,s=26,color=BLUE,edgecolor='white',lw=0.7,zorder=5,label='train'); ax.scatter(xs,ys,s=28,marker='s',facecolor='white',edgecolor=GREY,lw=1.2,zorder=5,label='test')
ax.legend(fontsize=8.5,loc='upper left',handletextpad=0.2,borderaxespad=0.2); ax.set_xlim(-0.3,6.4); ax.set_ylim(0,9); fin(ax,'$x$','$y$'); save(fig,'p_6A1.png')
# ---- 6A.2 bias-variance by hand
fl=np.array([2.,4,4,6]); st=np.array([3.5,4,4,4.5]); tr_=5.0; s2=0.5
def bv(p): b2=(p.mean()-tr_)**2; v=p.var(); return {'bias2':float(b2),'var':float(v),'total':float(s2+b2+v),'direct':float(s2+np.mean((tr_-p)**2))}
A['6A.2']={'flexible':bv(fl),'stiff':bv(st)}
fig,ax=small(); ax.axvline(tr_,color=GREY,lw=2.0,zorder=1); ax.text(tr_+0.08,2.2,'true $f$ = 5',fontsize=8.5,color=GREY,va='center')
ax.scatter(fl,np.full(4,1.4),s=30,color=GREEN,edgecolor='white',lw=0.7,zorder=5); ax.scatter(st,np.full(4,0.5),s=30,marker='s',color=OR,edgecolor='white',lw=0.7,zorder=5)
ax.text(1.75,1.85,'flexible model',fontsize=8.5,color=GREEN); ax.text(1.75,0.95,'stiff model',fontsize=8.5,color=OR); ax.set_xlim(1.5,7); ax.set_ylim(0,2.6); ax.set_yticks([]); fin(ax,'prediction $\\hat{f}$ at $x_0$'); ax.spines['left'].set_visible(False); save(fig,'p_6A2.png')
# ---- 6A.3 gap formula
p,s2b=6,0.09; A['6A.3']={'train30':s2b*(1-p/30),'test30':s2b*(1+p/30),'gap30':2*p*s2b/30,'n_for_0.006':2*p*s2b/0.006}
fig,ax=small(); nn=np.linspace(15,220,200); ax.plot(nn,2*p*s2b/nn,color=RED,lw=1.6); ax.axhline(0.006,color=GREY,lw=0.9,ls='--'); ax.plot(30,2*p*s2b/30,'o',color=BLUE,ms=5,zorder=5)
ax.text(34,0.04,'$n$ = 30',fontsize=8.5,color=BLUE); ax.text(215,0.0085,'0.006',fontsize=8.5,color=GREY,ha='right'); ax.set_ylim(0,0.075); ax.set_xlim(10,225); fin(ax,'training samples $n$','gap'); save(fig,'p_6A3.png')
# ---- 6A.4 ridge by hand
xr=np.array([-2.,-1,1,2]); yr=np.array([-3.,-1,1,3]); Sxy=float((xr*yr).sum()); Sxx=float((xr*xr).sum())
wa=lambda a: Sxy/(Sxx+a); A['6A.4']={'Sxy':Sxy,'Sxx':Sxx,'w0':wa(0),'w4':wa(4),'w10':wa(10),'mse4':float(np.mean((yr-wa(4)*xr)**2)),'alpha_for_0.5':Sxy/0.5-Sxx}
fig,ax=small(); tt=np.array([-2.6,2.6])
for a,c,ls in ((0,GREEN,'-'),(4,PUR,'--'),(10,PUR,':')): ax.plot(tt,wa(a)*tt,color=c,lw=1.4,ls=ls,label=f'$\\alpha$ = {a}')
ax.scatter(xr,yr,s=28,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.legend(fontsize=8.5,loc='upper left',handlelength=1.8,borderaxespad=0.2); ax.set_xlim(-2.6,2.6); ax.set_ylim(-4.2,4.2); fin(ax,'$x$','$y$'); save(fig,'p_6A4.png')
# ---- 6B.1 degree sweep with 100 samples
rng=np.random.default_rng(7)
def draw(n): x=np.sort(rng.uniform(0,1,n)); return x,np.sin(2*np.pi*x)+rng.normal(0,0.25,n)
x1,y1=draw(100); x2,y2=draw(400); rm=lambda a,b: float(np.sqrt(np.mean((a-b)**2))); T1=[];T2=[]
for d in range(15): g=Polynomial.fit(x1,y1,d); T1.append(rm(y1,g(x1))); T2.append(rm(y2,g(x2)))
ib=int(np.argmin(T2)); A['6B.1']={'best_degree':ib,'train_at_best':T1[ib],'test_at_best':T2[ib],'test_deg14':T2[14],'train_deg14':T1[14],'test_deg3':T2[3]}
fig,ax=small(); ax.plot(range(15),T1,'o-',color=BLUE,ms=2.5,lw=1.3,label='train'); ax.plot(range(15),T2,'s--',color=GREY,ms=2.5,lw=1.2,label='test'); ax.axhline(0.25,color=RED,lw=0.8,ls=':')
ax.set_ylim(0.15,0.8); ax.legend(fontsize=8.5,loc='upper right',handlelength=1.6,borderaxespad=0.2); fin(ax,'degree','RMSE'); save(fig,'p_6B1.png')
# ---- 6B.3 noise floor of learning curve
A['6B.3']={}; fig,ax=small(); sizes=np.array([20,40,80,160,240])
for noise,c,mk in ((0.25,BLUE,'o-'),(0.5,OR,'s--')):
    r2=np.random.default_rng(61); X=r2.uniform(0,1,(400,1)); y=np.sin(2*np.pi*X[:,0])+r2.normal(0,noise,400)
    m=make_pipeline(PolynomialFeatures(3,include_bias=False),LinearRegression())
    nu,a,b=learning_curve(m,X,y,train_sizes=sizes,cv=5,shuffle=True,random_state=0,scoring='neg_root_mean_squared_error')
    A['6B.3'][str(noise)]={'n':nu.tolist(),'train':(-a.mean(1)).round(3).tolist(),'val':(-b.mean(1)).round(3).tolist()}
    ax.plot(nu,-b.mean(1),mk,color=c,ms=2.5,lw=1.3,label=f'noise {noise}'); ax.axhline(noise,color=c,lw=0.7,ls=':')
ax.set_ylim(0.2,0.65); ax.legend(fontsize=8.5,loc='center right',handlelength=1.6,borderaxespad=0.2); fin(ax,'training samples $n$','validation RMSE'); save(fig,'p_6B3.png')
# ---- 6B.2 winner's curse against number of rules
def rounds(nm,nt,reps=200,seed=5):
    r3=np.random.default_rng(seed); b=[];fr=[]
    for _ in range(reps):
        yv=r3.integers(0,2,nt); Xv=r3.normal(size=(nt,nm)); acc=np.mean((Xv>0)==yv[:,None],axis=0); j=np.argmax(acc); b.append(acc[j])
        fr.append(np.mean((r3.normal(size=2000)>0)==r3.integers(0,2,2000)))
    return float(np.mean(b)),float(np.mean(fr))
ms=[1,10,100,1000]; res=[rounds(m,100) for m in ms]; A['6B.2']={str(m):{'best_test':round(r[0],3),'fresh':round(r[1],3)} for m,r in zip(ms,res)}
fig,ax=small(); ax.plot(ms,[r[0] for r in res],'o-',color=RED,ms=3,lw=1.4,label='best on test'); ax.plot(ms,[r[1] for r in res],'s--',color=GREEN,ms=3,lw=1.4,label='fresh data'); ax.set_xscale('log'); ax.set_ylim(0.45,0.75)
ax.legend(fontsize=8.5,loc='upper left',handlelength=1.6,borderaxespad=0.2); fin(ax,'number of rules','accuracy'); save(fig,'p_6B2.png')
# ---- 6C.1 diagnose a learning curve (degree 9, small n)
r4=np.random.default_rng(61); X=r4.uniform(0,1,(300,1)); y=np.sin(2*np.pi*X[:,0])+r4.normal(0,0.25,300); sz=np.array([15,20,30,50,80])
m=make_pipeline(PolynomialFeatures(9,include_bias=False),LinearRegression()); nu,a,b=learning_curve(m,X,y,train_sizes=sz,cv=5,shuffle=True,random_state=0,scoring='neg_root_mean_squared_error')
A['6C.1']={'n':nu.tolist(),'train':(-a.mean(1)).round(3).tolist(),'val':(-b.mean(1)).round(3).tolist()}
fig,ax=small(); ax.plot(nu,-a.mean(1),'o-',color=BLUE,ms=3,lw=1.4,label='training'); ax.plot(nu,-b.mean(1),'s--',color=GREY,ms=3,lw=1.4,label='validation'); ax.set_ylim(0.1,0.55)
ax.legend(fontsize=8.5,loc='upper right',handlelength=1.6,borderaxespad=0.2); fin(ax,'training samples $n$','RMSE'); save(fig,'p_6C1.png')
# ---- 6C.2 tuning on the test set
r5=np.random.default_rng(66); Xa=r5.uniform(0,1,(60,1)); ya=np.sin(2*np.pi*Xa[:,0])+r5.normal(0,0.25,60); Xb=r5.uniform(0,1,(15,1)); yb=np.sin(2*np.pi*Xb[:,0])+r5.normal(0,0.25,15)
Xc=r5.uniform(0,1,(5000,1)); yc=np.sin(2*np.pi*Xc[:,0])+r5.normal(0,0.25,5000); al=np.logspace(-6,1,30); T=[];F=[]
for a_ in al:
    mm=make_pipeline(PolynomialFeatures(9,include_bias=False),StandardScaler(),Ridge(alpha=a_)).fit(Xa[:30],ya[:30]); T.append(rm(yb,mm.predict(Xb))); F.append(rm(yc,mm.predict(Xc)))
ib=int(np.argmin(T)); A['6C.2']={'best_test':T[ib],'fresh_same_alpha':F[ib],'best_fresh':float(min(F)),'alpha_best_test':float(al[ib])}
fig,ax=small(); ax.plot(al,T,'s--',color=GREY,ms=2.5,lw=1.0,label='15 test samples'); ax.plot(al,F,'-',color=GREEN,lw=1.4,label='fresh data'); ax.plot(al[ib],T[ib],'o',mfc='none',mec=RED,ms=8,mew=1.4); ax.set_xscale('log'); ax.set_ylim(0.1,0.65)
ax.legend(fontsize=8.5,loc='upper left',handlelength=1.6,borderaxespad=0.2); fin(ax,'$\\alpha$','RMSE'); save(fig,'p_6C2.png')
# ---- 6C.3 doubling n
nn=np.linspace(10,200,200); s2=0.0625; fig,ax=small(); ax.plot(nn,s2*(1+4/nn),color=GREY,lw=1.5); ax.plot(nn,s2*(1-4/nn),color=BLUE,lw=1.5); ax.axhline(s2,color=RED,lw=0.9,ls=':')
ax.plot([40,80],[s2*(1+4/40),s2*(1+4/80)],'o',color=PUR,ms=4,zorder=5); ax.text(86,0.049,'train',fontsize=8.5,color=BLUE); ax.text(86,0.074,'test',fontsize=8.5,color=GREY); ax.set_ylim(0.03,0.1); ax.set_xlim(0,205); fin(ax,'training samples $n$','error'); save(fig,'p_6C3.png')
A['6C.3']={'test_n40':s2*(1+4/40),'test_n80':s2*(1+4/80),'gap_n40':2*4*s2/40,'gap_n80':2*4*s2/80}
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

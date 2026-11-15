# Problem figures of Chapter 9. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.linear_model import LinearRegression
A={}
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10])
def sc(ax,xx,yy,s=30): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def small(w=3.0,h=2.1): return plt.subplots(figsize=(w,h),dpi=240)
# 9A.1
xa=np.array([0.,1,2,3]); ya=np.array([1.,3,4,6]); m=LinearRegression().fit(xa[:,None],ya); pa=m.predict(xa[:,None])
A['9A.1']={'w':round(float(m.coef_[0]),3),'b':round(float(m.intercept_),3),'SSE':round(float(((ya-pa)**2).sum()),3)}
A['9A.4']={'SST':float(((ya-ya.mean())**2).sum()),'R2':round(float(m.score(xa[:,None],ya)),4)}
fig,ax=small(); sc(ax,xa,ya,40)
for a,b in zip(xa,ya): ax.text(a+0.12,b-0.55,f'({a:.0f}, {b:.0f})',fontsize=8,color=GREY)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xlim(-0.4,3.9); ax.set_ylim(0,7); save(fig,'p_9A1.png')
# 9A.2
p=1+1.6*x; A['9A.2']={'SSE':round(float(((y-p)**2).sum()),2)}
fig,ax=small(); xx=np.array([0,5.6]); ax.plot(xx,1+1.6*xx,color=OR,lw=1.6,label='$\\hat y = 1 + 1.6x$'); ax.plot(xx,0.9+1.7*xx,color=GREEN,lw=1.6,ls='--',label='least squares')
for a,b,c in zip(x,y,p): ax.plot([a,a],[b,c],color=RED,lw=1.2)
sc(ax,x,y); ax.legend(fontsize=8,loc='upper left'); ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); save(fig,'p_9A2.png')
# 9A.3
w0=float((x*y).sum()/(x*x).sum()); A['9A.3']={'sum_xy':float((x*y).sum()),'sum_xx':float((x*x).sum()),'w':round(w0,4),'SSE':round(float(((y-w0*x)**2).sum()),3),
  'sk':round(float(LinearRegression(fit_intercept=False).fit(x[:,None],y).coef_[0]),4)}
fig,ax=small(); ax.plot(xx,w0*xx,color=GREEN,lw=1.6); ax.plot(0,0,'o',color=GREEN,ms=6,clip_on=False,zorder=6); sc(ax,x,y); ax.text(2.4,0.6,'forced through\nthe origin',fontsize=8,color=GREEN,va='bottom',ha='left')
ax.set_xlim(0,5.6); ax.set_ylim(0,11.5); ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); save(fig,'p_9A3.png')
# 9A.4
fig,ax=small(); ax.axhline(ya.mean(),color=OR,lw=1.2,ls='--'); ax.plot([-0.3,3.5],m.predict([[-0.3],[3.5]]),color=GREEN,lw=1.6)
for a,b,c in zip(xa,ya,pa): ax.plot([a-0.05,a-0.05],[b,ya.mean()],color=OR,lw=1.2); ax.plot([a+0.05,a+0.05],[b,c],color=RED,lw=1.6)
sc(ax,xa,ya); ax.text(3.45,ya.mean()-0.5,'mean $\\bar y$',color=OR,fontsize=8,ha='right'); ax.text(0.1,6.1,'orange: SST parts\nred: SSE parts',fontsize=8,color=GREY,va='top'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_ylim(0,7); save(fig,'p_9A4.png')
# 9B.1  score schematic
fig,ax=new(3.4,1.5); box(ax,0,0.9,1.3,0.5,'X, y',fc=LGREY,ec=GREY); box(ax,1.9,0.9,1.5,0.5,'MyOLS.fit',fc=LB,ec=BLUE); box(ax,1.9,0.1,1.5,0.5,'.score(X, y)',fc='#F1EAF7',ec=PUR); box(ax,0,0.1,1.3,0.5,'$R^2$',fc='#E3F4EA',ec=GREEN,fs=10)
arr(ax,(1.3,1.15),(1.9,1.15),c=NAVY); arr(ax,(2.65,0.9),(2.65,0.6),c=NAVY); arr(ax,(1.9,0.35),(1.3,0.35),c=NAVY); ax.set_xlim(-0.1,3.5); ax.set_ylim(0,1.5); save(fig,'p_9B1.png')
# 9B.2 fan of lines
fig,ax=small(); A['9B.2']={}
for v,c in zip([10,15,20,25,30],plt.cm.Purples(np.linspace(0.35,0.95,5))):
    y2=y.copy(); y2[4]=v; mm=LinearRegression().fit(x[:,None],y2); ax.plot(xx,mm.predict(xx[:,None]),color=c,lw=1.4); ax.plot(5,v,'o',color=c,ms=5,zorder=6); A['9B.2'][str(v)]=[round(float(mm.coef_[0]),2),round(float(mm.intercept_),2)]
sc(ax,x[:4],y[:4]); ax.text(0.3,30,'$y_5$ = 10, 15, ..., 30',fontsize=8,color=PUR,ha='left',va='center'); ax.set_xlim(0,5.6); ax.set_ylim(-4,34); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); save(fig,'p_9B2.png')
# 9B.3 duplicated column
X2=np.c_[x,x]; mm=LinearRegression().fit(X2,y); A['9B.3']={'coef':[round(float(v),4) for v in mm.coef_],'rank':int(mm.rank_),'b':round(float(mm.intercept_),4)}
try:
    Xc=X2-X2.mean(0); np.linalg.solve(Xc.T@Xc,Xc.T@(y-y.mean())); A['9B.3']['normal']='no error'
except np.linalg.LinAlgError as e: A['9B.3']['normal']='LinAlgError: '+str(e)
fig,ax=new(3.2,1.9)
for j,(lab,c) in enumerate([('$x_1$',BLUE),('$x_2 = x_1$',BLUE),('$y$',GREEN)]):
    x0=[0.2,1.0,2.4][j]
    for i,v in enumerate([x,x,y][j]): ax.add_patch(Rectangle((x0,1.5-(i+1)*0.28),0.6,0.28,fc=LB if j<2 else '#E3F4EA',ec=c,lw=0.7)); ax.text(x0+0.3,1.5-(i+.5)*0.28,f'{v:.0f}',ha='center',va='center',fontsize=8)
    ax.text(x0+0.3,1.6,lab,ha='center',fontsize=9,color=c)
ax.text(0.9,-0.08,'two identical columns',fontsize=8,color=RED,ha='center'); ax.set_xlim(0,3.2); ax.set_ylim(-0.15,1.85); save(fig,'p_9B3.png')
# 9C.1 residual bars
r=y-(0.9+1.7*x); fig,ax=small(3.0,1.8); ax.axhline(0,color=GREEN,lw=1.2); ax.bar(x,r,width=0.5,color=[RED if v<0 else BLUE for v in r]); 
for a,v in zip(x,r): ax.text(a,v+(0.07 if v>=0 else -0.07),f'{v:+.1f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8)
ax.set_ylim(-1.05,0.95); ax.set_xlabel('$x$'); ax.set_ylabel('residual $r_i$'); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True); save(fig,'p_9C1.png')
A['9C.1']={'sum_r':round(float(r.sum()),10),'sum_xr':round(float((x*r).sum()),10),'nointercept_sum_r':round(float((y-w0*x).sum()),3)}
# 9C.2 train/test R2 vs p
rng=np.random.default_rng(7); n=60; Xa=rng.normal(size=(2000+n,58)); ya_=2*Xa[:,0]-Xa[:,1]+rng.normal(0,2,2000+n); ps=list(range(2,59,4)); tr=[];te=[]
for pp in ps:
    mm=LinearRegression().fit(Xa[:n,:pp],ya_[:n]); tr.append(mm.score(Xa[:n,:pp],ya_[:n])); te.append(mm.score(Xa[n:,:pp],ya_[n:]))
A['9C.2']={'p':ps,'train':[round(float(v),2) for v in tr],'test':[round(float(v),2) for v in te]}
fig,ax=small(3.2,2.1); ax.plot(ps,tr,'-o',color=BLUE,ms=3,lw=1.4,label='training $R^2$'); ax.plot(ps,te,'-o',color=GREY,ms=3,lw=1.4,label='test $R^2$'); ax.axvline(60,color=RED,lw=0.8,ls='--'); ax.text(59,-0.85,'$p = n$',color=RED,fontsize=8,ha='right')
ax.set_ylim(-1,1.08); ax.set_xlim(0,62); ax.set_xlabel('number of features $p$   ($n$ = 60)'); ax.set_ylabel('$R^2$'); ax.legend(fontsize=8,loc='lower left'); ax.grid(color=GRID,lw=0.5); save(fig,'p_9C2.png')
# 9C.3 units
mk=LinearRegression().fit(x[:,None],y); mmtr=LinearRegression().fit(1000*x[:,None],y); A['9C.3']={'w_km':round(float(mk.coef_[0]),4),'w_m':round(float(mmtr.coef_[0]),6),'b_m':round(float(mmtr.intercept_),4),'same_pred':bool(np.allclose(mk.predict(x[:,None]),mmtr.predict(1000*x[:,None])))}
fig,axs=plt.subplots(1,2,figsize=(3.6,1.8),dpi=240,sharey=True)
for ax,f,l in zip(axs,[1,1000],['distance (km)','distance (m)']): ax.plot(f*xx,0.9+1.7*xx,color=GREEN,lw=1.5); sc(ax,f*x,y,22); ax.set_xlabel(l,fontsize=8.5); ax.tick_params(labelsize=7.5)
axs[0].set_ylabel('time (min)',fontsize=8.5); fig.tight_layout(w_pad=0.5); save(fig,'p_9C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

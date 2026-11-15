# Problem figures and answers of Chapter 1. Run: python3 fig/probs.py
import sys,os,json,runpy; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.datasets import make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5})
A={}
def small(w=2.2,h=1.15): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 1A.1 reading a table
X=np.array([[2,5,1],[3,4,0],[7,1,2],[4,6,3]]); yv=np.array([10,9,20,18])
A['1A.1']={'shape':list(X.shape),'x23':int(X[1,2]),'x3':X[2].tolist(),'mean_col2':float(X[:,1].mean())}
fig,ax=new(2.16,1.37)
for j,h in enumerate(['$x_1$','$x_2$','$x_3$','$y$']): ax.text(0.2+j*0.55+0.27,1.55,h,ha='center',fontsize=9,color=BLUE if j<3 else GREEN)
for i in range(4):
    for j in range(4):
        v=X[i,j] if j<3 else yv[i]; ax.add_patch(Rectangle((0.2+j*0.55,1.25-(i+1)*0.3),0.55,0.3,fc=LB if j<3 else '#E3F4EA',ec=BLUE if j<3 else GREEN,lw=0.7)); ax.text(0.2+j*0.55+0.27,1.25-(i+.5)*0.3,str(v),ha='center',va='center',fontsize=8.5)
for i in range(4): ax.text(0.12,1.25-(i+.5)*0.3,str(i+1),ha='right',va='center',fontsize=8,color=GREY)
ax.set_xlim(-0.1,2.6); ax.set_ylim(-0.1,1.8); save(fig,'p_1A1.png')
# 1A.2 best constant
y7=np.array([1.,2,3,4,5,6,28]); A['1A.2']={'mean':float(y7.mean()),'median':float(np.median(y7)),'mse_mean':float(np.mean((y7-y7.mean())**2)),'mae_median':round(float(np.mean(np.abs(y7-np.median(y7)))),4),'mae_mean':round(float(np.mean(np.abs(y7-y7.mean()))),4)}
fig,ax=small(2.30,0.94); ax.scatter(y7,np.zeros(7),s=30,color=BLUE,edgecolor='white',zorder=5,clip_on=False); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('value'); ax.set_xlim(0,30); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_ylim(0,1); save(fig,'p_1A2.png')
# 1A.3 gradient descent
y3=np.array([1.,4,10]); c=0.; path=[c]
for _ in range(3): c=c-0.1*(-2*np.mean(y3-c)); path.append(c)
A['1A.3']={'mean':float(y3.mean()),'path':[round(float(v),4) for v in path]}
fig,ax=small(2.16,1.08); cc=np.linspace(-1,11,200); J=lambda c:np.mean((y3[None,:]-np.atleast_1d(c)[:,None])**2,axis=1); ax.plot(cc,J(cc),color=RED,lw=1.6); pp=np.array(path); ax.plot(pp,J(pp),'-o',color=GREEN,ms=4); ax.set_xlabel('$c$'); ax.set_ylabel('$J(c)$'); gr(ax); save(fig,'p_1A3.png')
# 1A.4 two scorecards
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10]); p=1+1.5*x; r=y-p
A['1A.4']={'r':r.tolist(),'SSE':float((r**2).sum()),'MSE':float(np.mean(r**2)),'MAE':float(np.mean(np.abs(r)))}
fig,ax=small(2.16,1.08); xx=np.array([0,5.6]); ax.plot(xx,1+1.5*xx,color=GREEN,lw=1.6)
for a,b_,c_ in zip(x,y,p): ax.plot([a,a],[b_,c_],color=RED,lw=1.4)
ax.scatter(x,y,s=28,color=BLUE,edgecolor='white',zorder=5); ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); gr(ax); save(fig,'p_1A4.png')
# 1B.1 add 100
cn=runpy.run_path('code/constants.py'); y6=np.append(cn['y'],100.0); g=np.round(np.arange(0.0,30.0,0.01),2)
Jsq=[np.mean((y6-c)**2) for c in g]; Jab=[np.mean(np.abs(y6-c)) for c in g]
A['1B.1']={'mean':round(float(y6.mean()),3),'median':float(np.median(y6)),'grid_sq':float(g[int(np.argmin(Jsq))]),'grid_abs':float(g[int(np.argmin(Jab))]),'abs_min_value':round(float(min(Jab)),3),'abs_at_4':round(float(np.mean(np.abs(y6-4))),3),'abs_at_5':round(float(np.mean(np.abs(y6-5))),3)}
fig,ax=small(2.16,1.08); cc=np.linspace(0,110,300); ax.plot(cc,[np.mean(np.abs(y6-c)) for c in cc],color=RED,lw=1.6); ax.scatter(y6,np.zeros(6),s=24,color=BLUE,edgecolor='white',zorder=5,clip_on=False); ax.set_xlabel('$c$'); ax.set_ylabel('mean absolute loss'); ax.set_ylim(0,60); gr(ax); save(fig,'p_1B1.png')
# 1B.2 learning rates
ds=runpy.run_path('code/descent.py'); A['1B.2']={str(e):[round(float(v),3) for v in ds['descend'](e)] for e in (0.5,0.75,1.0)}
fig,ax=small(2.16,1.08)
for e,cl,ls in zip((0.5,0.75,1.0),(GREEN,PUR,RED),('-','--',':')): ax.plot(ds['descend'](e),ls+'o',color=cl,ms=3,lw=1.3,label=f'$\\eta$ = {e}')
ax.axhline(6,color=GREY,lw=0.8); ax.set_xlabel('step $t$'); ax.set_ylabel('$c_t$'); ax.legend(fontsize=8,loc='center right',ncol=1); gr(ax); save(fig,'p_1B2.png')
# 1B.3 train/test on make_regression
Xr,yr=make_regression(n_samples=40,n_features=10,n_informative=3,noise=40.0,random_state=0); Xtr,Xte,ytr,yte=train_test_split(Xr,yr,test_size=0.5,random_state=0)
d=DummyRegressor().fit(Xtr,ytr); l=LinearRegression().fit(Xtr,ytr)
A['1B.3']={'dummy_train':round(d.score(Xtr,ytr),3),'dummy_test':round(d.score(Xte,yte),3),'lin_train':round(l.score(Xtr,ytr),3),'lin_test':round(l.score(Xte,yte),3)}
fig,ax=small(2.16,1.08); vals=[A['1B.3'][k] for k in ('dummy_train','dummy_test','lin_train','lin_test')]
ax.bar([0,1,3,4],vals,color=[BLUE,GREY,BLUE,GREY]); ax.set_xticks([0.5,3.5]); ax.set_xticklabels(['Dummy','Linear']); ax.set_ylabel('$R^2$ (blue train, grey test)',fontsize=8); ax.axhline(0,color=NAVY,lw=0.7); gr(ax); save(fig,'p_1B3.png')
# 1C.1 even n
y4=np.array([2.,3,5,9]); fig,ax=small(2.16,1.08); cc=np.linspace(0,12,300); ax.plot(cc,[np.mean(np.abs(y4-c)) for c in cc],color=RED,lw=1.6); ax.scatter(y4,np.zeros(4),s=26,color=BLUE,edgecolor='white',zorder=5,clip_on=False); ax.set_xlabel('$c$'); ax.set_ylabel('mean absolute loss'); ax.set_ylim(0,6); gr(ax); save(fig,'p_1C1.png')
A['1C.1']={'flat_value':float(np.mean(np.abs(y4-4)))}
# 1C.2 training error only
gn=runpy.run_path('code/generalise.py'); P=gn['Polynomial']; rm=gn['rmse']; dg=[0,1,2,3,5,7,9]; tre=[rm(gn['y_tr'],P.fit(gn['x_tr'],gn['y_tr'],k)(gn['x_tr'])) for k in dg]
fig,ax=small(2.16,1.08); ax.plot(dg,tre,'-o',color=BLUE,ms=3.5,lw=1.4); ax.set_xlabel('degree'); ax.set_ylabel('training RMSE'); gr(ax); save(fig,'p_1C2.png')
A['1C.2']={'train_rmse':[round(float(v),3) for v in tre]}
# 1C.3 clustering without labels
Xk,_=make_blobs(n_samples=75,centers=[(-3,-1),(0,3),(3,-1)],cluster_std=0.85,random_state=4)
fig,axs=plt.subplots(1,2,figsize=(2.8,1.2),dpi=240,sharey=True)
for ax,k in zip(axs,(2,3)):
    lab=KMeans(n_clusters=k,n_init=10,random_state=0).fit_predict(Xk); mk=['o','s','^']; cl=[BLUE,OR,NAVY]
    for q in range(k): ax.scatter(Xk[lab==q,0],Xk[lab==q,1],s=14,color=cl[q],marker=mk[q],edgecolor='white',lw=0.4)
    ax.set_title(f'{k} groups',fontsize=8.5,color=NAVY); ax.tick_params(labelsize=7.5); gr(ax)
fig.tight_layout(w_pad=0.5); save(fig,'p_1C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A,indent=0))

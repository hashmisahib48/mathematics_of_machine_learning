# Problem figures of Chapter 20. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Circle, Rectangle
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.datasets import make_moons, load_wine, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
A={}
def small(w=2.5,h=1.45): return plt.subplots(figsize=(w,h),dpi=240)
def ax_(ax,xl='',yl='',g=True):
    ax.set_xlabel(xl,fontsize=9); ax.set_ylabel(yl,fontsize=9); ax.tick_params(labelsize=8.5)
    if g: ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 20A.1 distances to two points
P0=np.array([0.,0]); Pa=np.array([3.,3]); Pb=np.array([4.,0])
dd=lambda p,qq:(float((np.abs(p)**qq).sum()**(1/qq)) if qq!=np.inf else float(np.abs(p).max()))
A['20A.1']={str(qq):{'A':round(dd(Pa,qq),3),'B':round(dd(Pb,qq),3)} for qq in (1,2,np.inf)}
fig,ax=small(2.5,1.9); th=np.linspace(0,2*np.pi,400)
ax.plot([4,0,-4,0,4],[0,4,0,-4,0],color=RED,lw=1.2); ax.plot(4*np.cos(th),4*np.sin(th),color=GREEN,lw=1.2); ax.plot([-4,4,4,-4,-4],[-4,-4,4,4,-4],color=OR,lw=1.2,ls=':')
ax.scatter(*Pa,s=36,color=BLUE,zorder=5); ax.scatter(*Pb,s=36,color=BLUE,marker='s',zorder=5); ax.scatter(0,0,s=60,marker='*',color=NAVY,zorder=6)
ax.text(3.2,3.25,'A',fontsize=9); ax.text(4.2,-0.9,'B',fontsize=9); ax.set_aspect('equal'); ax.set_xlim(-5,5.5); ax.set_ylim(-5,5); ax_(ax,'$x_1$','$x_2$'); ax.set_xticks([-4,0,4]); ax.set_yticks([-4,0,4]); save(fig,'p_20A1.png')
# 20A.2 a vote on a line
x1=np.array([1.,2,4,5,7,8]); y1=np.array([0,0,1,0,1,1]); x0=4.6; d=np.abs(x1-x0); o=np.argsort(d,kind='stable')
res={}
for k in (1,2,3,4,5):
    v=np.bincount(y1[o[:k]],minlength=2); res[str(k)]={'votes':v.tolist(),'pred':int(KNeighborsClassifier(k).fit(x1[:,None],y1).predict([[x0]])[0])}
w=1/d[o[:3]]; sc=[float(w[y1[o[:3]]==c].sum()) for c in (0,1)]; res['w3']={'score':[round(v,4) for v in sc],'pred':int(KNeighborsClassifier(3,weights='distance').fit(x1[:,None],y1).predict([[x0]])[0])}
A['20A.2']=res
fig,ax=small(2.6,1.15); ax.axhline(0,color=GRID,lw=1)
for a_,b_ in zip(x1,y1): ax.scatter(a_,0,s=40,color=BLUE if b_==0 else OR,marker='o' if b_==0 else 's',zorder=4,edgecolor='white')
ax.scatter(x0,0,s=90,marker='*',color=NAVY,zorder=5); ax.text(x0,0.35,'query',fontsize=9,ha='center',color=NAVY); ax.set_ylim(-0.6,1); ax.set_yticks([]); ax.set_xlim(0.3,8.7); ax_(ax,'$x$',g=False); ax.spines['left'].set_visible(False); save(fig,'p_20A2.png')
# 20A.3 regression by hand
x2=np.array([1.,2,4,7]); y2=np.array([2.,3,7,9]); x0=3.5; d=np.abs(x2-x0); o=np.argsort(d,kind='stable'); res={}
for k in (2,3):
    nn=o[:k]; res[f'k{k}_uniform']=round(float(y2[nn].mean()),4); w=1/d[nn]; res[f'k{k}_weighted']=round(float((w*y2[nn]).sum()/w.sum()),4)
    for wt,key in (('uniform','u'),('distance','w')):
        assert abs(KNeighborsRegressor(k,weights=wt).fit(x2[:,None],y2).predict([[x0]])[0]-res[f'k{k}_'+('uniform' if key=='u' else 'weighted')])<1e-4
A['20A.3']=res
fig,ax=small(2.5,1.45); ax.scatter(x2,y2,s=34,color=BLUE,zorder=4,edgecolor='white'); ax.axvline(x0,color=PUR,lw=1,ls='--'); ax.text(x0+0.15,1.0,'query\n$x_0=3.5$',fontsize=8.5,color=PUR); ax_(ax,'$x$','$y$'); ax.set_xlim(0,8); ax.set_ylim(0,10.5); save(fig,'p_20A3.png')
# 20B.1 regressor from scratch
rng=np.random.default_rng(0); Xr,yr=make_regression(200,4,noise=5,random_state=0); Xq=rng.normal(size=(50,4))
def myknn_reg(Xt,yt,Xq_,k,weights):
    d2=((Xq_[:,None]-Xt[None])**2).sum(2); idx=np.argsort(d2,1,kind='stable')[:,:k]; dist=np.sqrt(np.take_along_axis(d2,idx,1))
    w=np.ones_like(dist) if weights=='uniform' else 1/dist
    return (w*yt[idx]).sum(1)/w.sum(1)
mx=max(float(np.abs(myknn_reg(Xr,yr,Xq,5,wt)-KNeighborsRegressor(5,weights=wt).fit(Xr,yr).predict(Xq)).max()) for wt in ('uniform','distance'))
A['20B.1']={'max_abs_diff':mx}
xs=np.linspace(0,6,40); ys=np.sin(xs)+np.random.default_rng(1).normal(0,0.25,40); xg=np.linspace(0,6,300); m=KNeighborsRegressor(5).fit(xs[:,None],ys)
fig,ax=small(); ax.scatter(xs,ys,s=12,color=BLUE,zorder=3); ax.plot(xg,m.predict(xg[:,None]),color=GREEN,lw=1.5); ax_(ax,'$x$','$y$'); save(fig,'p_20B1.png')
# 20B.2 scalers on wine
Xw,yw=load_wine(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xw,yw,test_size=0.25,random_state=0,stratify=yw)
acc={'raw':KNeighborsClassifier().fit(Xtr,ytr).score(Xte,yte)}
for nm,S in (('standard',StandardScaler),('minmax',MinMaxScaler)): acc[nm]=make_pipeline(S(),KNeighborsClassifier()).fit(Xtr,ytr).score(Xte,yte)
A['20B.2']={k:round(float(v),4) for k,v in acc.items()}
fig,ax=small(2.5,1.45); ax.bar(range(3),[acc['raw'],acc['standard'],acc['minmax']],color=[RED,GREEN,GREEN],width=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(['raw','standard','min-max'],fontsize=8.5); ax.set_ylim(0.5,1.0); ax_(ax,'','test accuracy'); ax.grid(axis='x',visible=False); save(fig,'p_20B2.png')
# 20B.3 radius classifier on moons
Xm,ym=make_moons(300,noise=0.3,random_state=0); Xa,Xb,ya,yb=train_test_split(Xm,ym,test_size=0.25,random_state=0,stratify=ym); res={}
rs=[0.1,0.2,0.4,0.8]
for r in rs:
    m=RadiusNeighborsClassifier(radius=r,outlier_label='most_frequent').fit(Xa,ya); nb=m.radius_neighbors(Xb,return_distance=False); res[str(r)]={'test_acc':round(float(m.score(Xb,yb)),4),'no_neighbour':int(sum(len(v)==0 for v in nb)),'mean_neighbours':round(float(np.mean([len(v) for v in nb])),1)}
A['20B.3']=res
fig,ax=small(); ax.plot(rs,[res[str(r)]['test_acc'] for r in rs],'-o',color=BLUE,ms=3.5,lw=1.4); ax_(ax,'radius $r$','test accuracy'); ax.set_xticks(rs); ax.tick_params(labelsize=8); save(fig,'p_20B3.png')
# 20C.1 best k against n
ks=list(range(1,62,2)); bestk=[]; ns=[50,100,200,400,800,1600]
for n in ns:
    M=[]
    for sd_ in range(12):
        Xq_,yq_=make_moons(n,noise=0.3,random_state=10+sd_); M.append([cross_val_score(KNeighborsClassifier(k),Xq_,yq_,cv=5).mean() for k in ks if k<=0.8*n])
    bestk.append(int(ks[int(np.argmax(np.mean(M,0)))]))
A['20C.1']={'n':ns,'best_k':bestk}
fig,ax=small(); ax.semilogx(ns,bestk,'-o',color=PUR,ms=3.5,lw=1.4); ax_(ax,'training set size $n$','best $k$'); ax.set_xticks([50,200,800,1600]); ax.set_xticklabels(['50','200','800','1600']); save(fig,'p_20C1.png')
# 20C.2 share of squared distance per feature
def share(Z):
    s=np.zeros(Z.shape[1]); n=0
    rng=np.random.default_rng(0)
    for _ in range(3000):
        i,j=rng.integers(0,len(Z),2)
        if i!=j: s+=(Z[i]-Z[j])**2; n+=1
    return s/s.sum()
sr=share(Xw); ss=share(StandardScaler().fit_transform(Xw)); A['20C.2']={'raw_max_share':round(float(sr.max()),4),'raw_argmax':int(sr.argmax()),'scaled_max_share':round(float(ss.max()),4)}
fig,ax=small(); ax.bar(np.arange(13)-0.2,sr,width=0.4,color=RED,label='raw'); ax.bar(np.arange(13)+0.2,ss,width=0.4,color=GREEN,label='standardised'); ax_(ax,'feature number','share of $d^2$'); ax.legend(fontsize=8,loc='upper left'); ax.set_xticks([0,4,8,12]); ax.set_xticklabels(['1','5','9','13']); save(fig,'p_20C2.png')
# 20C.3 operations per prediction
n_,p_=10**6,50; A['20C.3']={'knn_brute':n_*p_,'linear':p_}
fig,ax=small(); ax.bar([0,1],[n_*p_,p_],color=[RED,GREEN],width=0.55); ax.set_yscale('log'); ax.set_xticks([0,1]); ax.set_xticklabels(['k-NN\nbrute force','logistic\nregression'],fontsize=8.5); ax_(ax,'','operations'); ax.grid(axis='x',visible=False); ax.set_ylim(1,3e8); save(fig,'p_20C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

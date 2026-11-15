# Problem figures of Chapter 50. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from cars import make_cars
from toy import cars as toy
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.0,h=0.95): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*110,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 50A.1 one-hot by hand
cat=np.array(list('aabccc')); y=np.array([4,6,10,3,5,7.]); ks=np.unique(cat); D=(cat[:,None]==ks).astype(float)
A['50A.1']={'DtD':np.diag(D.T@D).tolist(),'rank_with_intercept':int(np.linalg.matrix_rank(np.hstack([np.ones((6,1)),D]))),'columns_with_intercept':4,'w':r4(D.T@y/D.sum(0)),'rank_drop_first':int(np.linalg.matrix_rank(np.hstack([np.ones((6,1)),D[:,1:]])))}
fig,ax=small(2.0,0.95); ax.imshow(np.hstack([D,y[:,None]/10]),cmap='Blues',vmin=0,vmax=1.3,aspect='auto')
for i in range(6):
    for j in range(3): ax.text(j,i,f'{int(D[i,j])}',ha='center',va='center',fontsize=9,color=NAVY if D[i,j]==0 else 'white')
    ax.text(3,i,f'{int(y[i])}',ha='center',va='center',fontsize=9,color=NAVY)
ax.set_xticks(range(4)); ax.set_xticklabels(['a','b','c','$y$']); ax.set_yticks(range(6)); ax.set_yticklabels([str(i+1) for i in range(6)]); ax.set_ylabel('row'); ax.spines[:].set_visible(False); ax.tick_params(length=0); sv(fig,'p_50A1.png')

# 50A.2 shrinkage by hand
n=np.array([1,4,20.]); mc=np.array([30,20,10.]); yb=14.; m=4.; t=(n*mc+m*yb)/(n+m)
A['50A.2']={'t':r4(t),'lambda':r4(n/(n+m)),'m':m,'ybar':yb}
fig,ax=small(2.0,0.95); xx=np.arange(3); ax.bar(xx-0.2,mc,0.38,color=BLUE,label='raw mean'); ax.bar(xx+0.2,t,0.38,color=GREEN,label='shrunk'); ax.axhline(yb,color=GREY,lw=0.8,ls='--')
ax.set_xticks(xx); ax.set_xticklabels(['$n=1$','$n=4$','$n=20$']); ax.set_ylim(0,52); ax.legend(loc='upper center',ncol=2,handlelength=1.0,fontsize=8.5,columnspacing=0.8); sv(fig,'p_50A2.png')

# data for B
df=make_cars(4000,0); tr,te=train_test_split(df,random_state=0)
# 50B.1 min_frequency
ks_=[1,5,10,20,50]; cols=[int(OneHotEncoder(min_frequency=k,sparse_output=False).fit(tr[['city']]).transform(tr[['city']]).shape[1]) for k in ks_]
inf=[int(len(OneHotEncoder(min_frequency=k,sparse_output=False).fit(tr[['city']]).infrequent_categories_[0])) if k>1 else 0 for k in ks_]
A['50B.1']={'min_frequency':ks_,'columns':cols,'infrequent_cities':inf}
fig,ax=small(2.0,0.95); ax.plot(ks_,cols,color=PUR,marker='o',ms=4,lw=1.2); ax.set_xscale('log'); ax.set_xticks(ks_); ax.set_xticklabels([str(k) for k in ks_]); ax.minorticks_off(); ax.set_xlabel('min_frequency'); ax.set_ylabel('columns'); grid(ax); sv(fig,'p_50B1.png')

# 50B.2 leakage against category size
sizes=[1,2,3,5,10,30]; c1=[];c2=[]
for r in sizes:
    k=1200//r; rng=np.random.RandomState(0); ids=rng.permutation(np.repeat(np.arange(k),r)).reshape(-1,1).astype(str); yy=rng.normal(size=len(ids))
    te_=TargetEncoder(smooth=5.0,target_type='continuous',cv=KFold(5,shuffle=True,random_state=0))
    Za=te_.fit(ids,yy).transform(ids)[:,0]; Zb=te_.fit_transform(ids,yy)[:,0]
    c1.append(float(np.corrcoef(Za,yy)[0,1])); c2.append(float(np.corrcoef(Zb,yy)[0,1]))
A['50B.2']={'rows_per_category':sizes,'corr_fit_then_transform':r4(c1),'corr_fit_transform':r4(c2)}
fig,ax=small(2.0,0.95); ax.plot(sizes,c1,color=RED,marker='o',ms=4,lw=1.2,label='fit, transform'); ax.plot(sizes,c2,color=GREEN,marker='s',ms=4,lw=1.2,label='fit_transform'); ax.set_xscale('log'); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes]); ax.minorticks_off(); ax.set_xlabel('rows per category'); ax.set_ylabel('corr$(Z,y)$'); ax.set_ylim(-0.55,1.5); ax.set_yticks([0,1]); ax.text(2.2,1.2,'fit, transform',color=RED,fontsize=8.5,va='center'); ax.text(2.0,-0.33,'fit_transform',color=GREEN,fontsize=8.5,va='center'); grid(ax); sv(fig,'p_50B2.png')

# 50C.1 ridge with and without dropping
Dt=D; yt=y
w_all=Ridge(1.0).fit(Dt,yt); w_dr=Ridge(1.0).fit(Dt[:,1:],yt)
A['50C.1']={'all':{'coef':r4(w_all.coef_),'intercept':round(float(w_all.intercept_),4)},'drop_first':{'coef':r4(w_dr.coef_),'intercept':round(float(w_dr.intercept_),4)},'categories':['a','b','c']}
fig,ax=small(2.0,0.95); xx=np.arange(3); ax.bar(xx-0.2,w_all.coef_,0.38,color=GREEN,label='all three'); ax.bar(xx[1:]+0.2,w_dr.coef_,0.38,color=OR,label='a dropped'); ax.axhline(0,color=GREY,lw=0.7); ax.set_xticks(xx); ax.set_xticklabels(['a','b','c']); ax.set_ylabel('coefficient'); ax.set_ylim(-3.2,5.2); ax.legend(loc='upper center',ncol=2,handlelength=1.0,fontsize=8.5,columnspacing=0.8); sv(fig,'p_50C1.png')

# 50C.2 cyclic feature
fig,ax=small(2.0,0.95); ax.set_aspect('equal'); ax.axis('off'); h=np.arange(24); th=2*np.pi*h/24
ax.scatter(np.sin(th),np.cos(th),s=9,color=BLUE,zorder=3)
for hh in (0,6,12,18): ax.text(1.28*np.sin(th[hh]),1.28*np.cos(th[hh]),str(hh),ha='center',va='center',fontsize=9,color=NAVY)
ax.set_xlim(-2.2,2.2); ax.set_ylim(-1.5,1.5); sv(fig,'p_50C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1)
json.dump(CHK,open('verify/problem_font_check.json','w'),indent=1)
print(json.dumps(A)[:1800]); print(CHK)

# Problem figures of Chapter 54. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="1"
import sys,json,warnings,tempfile; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
import pandas as pd
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from cars import make_cars
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.0): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 54A.1 retransformation by hand
mu_,s2=2.0,0.25; med=float(np.exp(mu_)); mean=float(np.exp(mu_+s2/2))
A['54A.1']={'median':round(med,3),'mean':round(mean,3),'factor':round(float(np.exp(s2/2)),4),'percent_higher':round(100*(mean/med-1),2)}
fig,ax=small(); sg=np.sqrt(s2); t=np.linspace(0.2,30,400); dens=np.exp(-(np.log(t)-mu_)**2/(2*s2))/(t*sg*np.sqrt(2*np.pi))
ax.plot(t,dens,color=BLUE,lw=1.4); ax.axvline(med,color=GREEN,lw=1.3,ls='--'); ax.axvline(mean,color=PUR,lw=1.3)
ax.text(med-0.3,0.172,'median',color=GREEN,fontsize=9,ha='right',va='center'); ax.text(mean+0.3,0.172,'mean',color=PUR,fontsize=9,ha='left',va='center')
ax.set_xlim(0,24); ax.set_ylim(0,0.19); ax.set_xlabel('$y$'); ax.set_yticks([]); ax.spines['left'].set_visible(False); sv(fig,'p_54A1.png')

# 54A.2 chance correlation
f=lambda n,p: float(np.sqrt(2*np.log(2*p)/n))
A['54A.2']={'n':100,'p1000':round(f(100,1000),3),'p10':round(f(100,10),3),'p_to_double':'(none: grows with ln p)','ratio_1000_to_10':round(f(100,1000)/f(100,10),3)}
fig,ax=small(); pp=np.logspace(0.5,4,60); ax.plot(pp,[f(100,p) for p in pp],color=RED,lw=1.4); ax.scatter([10,1000],[f(100,10),f(100,1000)],s=16,color=NAVY,zorder=3)
ax.set_xscale('log'); ax.set_xlabel('features $p$'); ax.set_ylabel('chance $|r|$'); ax.set_ylim(0.1,0.6); ax.minorticks_off(); ax.set_xticks([10,100,1000,10000]); ax.set_xticklabels(['10','100','1 000','10 000']); grid(ax); sv(fig,'p_54A2.png')

# 54A.3 counting columns
df=make_cars(2000,0); X=df.drop(columns='price'); res={}
for lab,rem,drop in (('drop',"drop",None),('passthrough','passthrough',None),('passthrough, drop first','passthrough','first')):
    ct=ColumnTransformer([('c',OneHotEncoder(drop=drop),['colour','fuel']),('n',StandardScaler(),['age'])],remainder=rem).fit(X)
    res[lab]=int(ct.transform(X).shape[1]) if not hasattr(ct.transform(X),'toarray') else int(ct.transform(X).shape[1])
A['54A.3']={'widths':res,'one_hot':{'colour':6,'fuel':4},'remainder_columns':['trim','city','km']}
fig,ax=small(); labs=list(res); vals=[res[k] for k in labs]; ax.barh(range(3),vals,color=[BLUE,GREEN,OR],height=0.6)
for i_,v in enumerate(vals): ax.text(v+0.3,i_,str(v),va='center',fontsize=9.5,color=NAVY)
ax.set_yticks(range(3)); ax.set_yticklabels(['(a)','(b)','(c)'],fontsize=9.5); ax.invert_yaxis(); ax.set_xlim(0,17); ax.set_xlabel('output columns'); sv(fig,'p_54A3.png')

# 54B.1 leak against the number of kept features
def one(seed,n=60,p=1000,k=10):
    rng=np.random.RandomState(seed); Xn=rng.normal(size=(n,p)); y=rng.randint(0,2,n); cv=StratifiedKFold(5,shuffle=True,random_state=0); clf=LogisticRegression(max_iter=1000)
    lk=cross_val_score(clf,SelectKBest(f_classif,k=k).fit_transform(Xn,y),y,cv=cv).mean(); hn=cross_val_score(make_pipeline(SelectKBest(f_classif,k=k),clf),Xn,y,cv=cv).mean(); return lk,hn
ks=[1,10,50,200]; out=[[one(s,k=k) for s in range(20)] for k in ks]
A['54B.1']={'k':ks,'leaky':r4([np.mean([a for a,b in o]) for o in out]),'honest':r4([np.mean([b for a,b in o]) for o in out]),'n_data_sets':20}
fig,ax=small(); ax.plot(ks,A['54B.1']['leaky'],color=RED,marker='o',ms=3.8,lw=1.3); ax.plot(ks,A['54B.1']['honest'],color=GREEN,marker='s',ms=3.8,lw=1.3,ls='--')
ax.axhline(0.5,color=GREY,lw=0.7,ls=':'); ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off(); ax.set_xlabel('features kept $k$'); ax.set_ylabel('accuracy'); ax.set_ylim(0.4,1.0)
ax.text(1.6,0.95,'selected first',color=RED,fontsize=9,va='center'); ax.text(1.5,0.6,'in the pipeline',color=GREEN,fontsize=9,va='center'); grid(ax); sv(fig,'p_54B1.png')

# 54B.2 counting fits with and without memory
class Counting(BaseEstimator,TransformerMixin):
    calls=0
    def fit(self,X,y=None):
        type(self).calls+=1; self.mean_=X.mean(0); return self
    def transform(self,X): return X-self.mean_
Xb,yb=load_breast_cancer(return_X_y=True); cnt={}
for lab,mem in (('no memory',None),('memory',Memory(tempfile.mkdtemp(),verbose=0))):
    Counting.calls=0
    for a in (0.1,1.0,10.0,100.0):
        cross_val_score(Pipeline([('c',Counting()),('r',Ridge(alpha=a))],memory=mem),Xb,yb,cv=KFold(5)) 
    cnt[lab]=Counting.calls
A['54B.2']={'fits_of_the_transformer':cnt,'alphas':4,'folds':5}
fig,ax=small(); ax.bar([0,1],[cnt['no memory'],cnt['memory']],color=[RED,GREEN],width=0.55)
for i_,v in enumerate([cnt['no memory'],cnt['memory']]): ax.text(i_,v+0.8,str(v),ha='center',fontsize=9.5,color=NAVY)
ax.set_xticks([0,1]); ax.set_xticklabels(['no memory','memory']); ax.set_ylabel('fits'); ax.set_ylim(0,26); sv(fig,'p_54B2.png')

# 54B.3 smearing
rng=np.random.RandomState(7); N=20000; x=rng.uniform(0,3,(N,1)); sig=0.5; y=np.exp(1+0.8*x[:,0]+rng.normal(0,sig,N))
t=TransformedTargetRegressor(LinearRegression(),func=np.log,inverse_func=np.exp).fit(x,y); pr=t.predict(x); sm=float(np.exp(np.log(y)-t.regressor_.predict(x)).mean())
A['54B.3']={'sigma':sig,'exp_s2_2':round(float(np.exp(sig**2/2)),4),'smearing_factor':round(sm,4),'mean_error_plain':round(float(np.mean(y-pr)),3),'mean_error_smeared':round(float(np.mean(y-pr*sm)),3),'rmse_plain':round(float(np.sqrt(np.mean((y-pr)**2))),3),'rmse_smeared':round(float(np.sqrt(np.mean((y-pr*sm)**2))),3)}
fig,ax=small(); ax.bar([0,1],[A['54B.3']['mean_error_plain'],A['54B.3']['mean_error_smeared']],color=[RED,GREEN],width=0.55); ax.axhline(0,color=GREY,lw=0.7)
ax.set_xticks([0,1]); ax.set_xticklabels(['plain','smeared']); ax.set_ylabel('mean error'); ax.set_ylim(-0.4,2.3)
for i_,v in enumerate([A['54B.3']['mean_error_plain'],A['54B.3']['mean_error_smeared']]): ax.text(i_,max(v,0)+0.12,f'{v:.3f}'.replace('-','$-$'),ha='center',fontsize=9,color=NAVY)
sv(fig,'p_54B3.png')

# 54C.1 a frozen scaler inside cross-validation
Xs,ys=load_breast_cancer(return_X_y=True); cv=StratifiedKFold(5,shuffle=True,random_state=0)
frozen=make_pipeline(FrozenEstimator(StandardScaler().fit(Xs)),SVC()); normal=make_pipeline(StandardScaler(),SVC())
a_f=cross_val_score(frozen,Xs,ys,cv=cv).mean(); a_n=cross_val_score(normal,Xs,ys,cv=cv).mean()
A['54C.1']={'frozen_on_all_rows':round(float(a_f),4),'refitted_per_fold':round(float(a_n),4)}
fig,ax=small(); ax.plot([a_f,a_n],[1,0],'o',color=NAVY,ms=5); ax.plot([a_f],[1],'o',color=RED,ms=6); ax.plot([a_n],[0],'o',color=GREEN,ms=6); ax.set_yticks([0,1]); ax.set_yticklabels(['refitted','frozen'],fontsize=9); ax.set_ylim(-0.6,1.6); ax.set_xlim(0.970,0.980); ax.set_xticks([0.970,0.975,0.980]); ax.set_xlabel('5-fold accuracy'); grid(ax)
ax.text(a_f,1.35,f'{a_f:.4f}',ha='center',fontsize=9,color=NAVY); ax.text(a_n,0.35,f'{a_n:.4f}',ha='center',fontsize=9,color=NAVY)
sv(fig,'p_54C1.png')

# 54C.2 imputing before the split
d=make_cars(2000,0).km.values.copy(); r=np.random.RandomState(5); mask=r.rand(2000)<0.08; d[mask]=np.nan
tr,te=d[:1500],d[1500:]; m_all=np.nanmean(d); m_tr=np.nanmean(tr)
A['54C.2']={'mean_all_rows':round(float(m_all),3),'mean_training_rows':round(float(m_tr),3),'mean_test_rows':round(float(np.nanmean(te)),3)}
fig,ax=small(); ax.bar([0,1,2],[m_tr,m_all,np.nanmean(te)],color=[GREEN,RED,GREY],width=0.6); ax.set_xticks([0,1,2]); ax.set_xticklabels(['train','all','test']); ax.set_ylabel('mean of km'); ax.set_ylim(0,max(m_tr,m_all,np.nanmean(te))*1.3)
for i_,v in enumerate([m_tr,m_all,np.nanmean(te)]): ax.text(i_,v+2,f'{v:.1f}',ha='center',fontsize=9,color=NAVY)
sv(fig,'p_54C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1)
json.dump(CHK,open('verify/problem_font_check.json','w'),indent=1)
print(json.dumps(A,indent=1)); print(CHK)

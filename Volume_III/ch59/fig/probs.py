# Problem figures of Chapter 59. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_breast_cancer, load_wine, make_regression, make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, KFold, cross_val_score, cross_validate
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
A={}; CHK={}
def small(w=2.2,h=1.1): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name,wpx=140):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*wpx,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# 59A.1 the bill of a search, counted
COUNT=[0]
class CountingRidge(RegressorMixin,BaseEstimator):
    def __init__(self,alpha=1.0): self.alpha=alpha
    def fit(self,X,y):
        COUNT[0]+=1; self.w_=np.linalg.solve(X.T@X+self.alpha*np.eye(X.shape[1]),X.T@y); return self
    def predict(self,X): return X@self.w_
Xr,yr=make_regression(200,5,noise=1.0,random_state=0)
srch=RandomizedSearchCV(CountingRidge(),{'alpha':loguniform(1e-3,1e3)},n_iter=40,cv=RepeatedKFold(n_splits=5,n_repeats=2,random_state=0),random_state=0)
srch.fit(Xr,yr); one=COUNT[0]; COUNT[0]=0
cross_val_score(srch,Xr,yr,cv=KFold(5)); nest=COUNT[0]
A['59A.1']={'fits_search':one,'fits_nested':nest}
fig,ax=small(2.2,1.05); ax.barh([1,0],[one,nest],color=[BLUE,GREEN],height=0.6,edgecolor=NAVY,lw=0.5)
for yy,v in ((1,one),(0,nest)): ax.text(v+40,yy,str(v),va='center',fontsize=7.8,color=NAVY)
ax.set_yticks([1,0]); ax.set_yticklabels(['search','nested']); ax.set_xlim(0,2700); ax.set_xticks([0,1000,2000]); ax.set_xlabel('fits'); grid(ax); sv(fig,'p_59A1.png')

# 59A.2 an interval for 131 of 143
n=143; e=12; a=1-e/n; se=np.sqrt(a*(1-a)/n); lo,hi=a-1.96*se,a+1.96*se
A['59A.2']={'accuracy':round(a,4),'se':round(se,4),'interval':[round(lo,3),round(hi,3)],'half_width':round(1.96*se,4)}
fig,ax=small(2.2,0.85); ax.plot([lo,hi],[0,0],color=GREEN,lw=3,solid_capstyle='butt'); ax.scatter([a],[0],s=26,color=NAVY,zorder=3)
ax.text(a,0.45,'%.3f'%a,ha='center',fontsize=7.8,color=NAVY); ax.text(lo,-0.62,'%.3f'%lo,ha='center',fontsize=7.8,color=NAVY); ax.text(hi,-0.62,'%.3f'%hi,ha='center',fontsize=7.8,color=NAVY)
ax.set_ylim(-1.0,0.9); ax.set_xlim(0.84,1.0); ax.set_yticks([]); ax.set_xticks([0.85,0.90,0.95,1.0]); ax.set_xlabel('accuracy'); ax.spines['left'].set_visible(False); grid(ax); sv(fig,'p_59A2.png')

# 59B.1 one-hot instead of target encoding for district
from case_b1 import Xtr as hX, ytr as hy, prep, hgb, log
from sklearn.preprocessing import OneHotEncoder
def mae(m): return float(-cross_val_score(m,hX,hy,cv=5,scoring='neg_mean_absolute_error').mean())
base=mae(log(hgb()))
pipe=log(hgb()); pipe.regressor.steps[0][1].set_params(district=OneHotEncoder(handle_unknown='ignore',sparse_output=False))
oh=mae(pipe)
A['59B.1']={'MAE_target_encoding':round(base,2),'MAE_one_hot':round(oh,2),'rows':len(hX),'districts':int(hX['district'].nunique()),'smallest_district_rows':int(hX['district'].value_counts().min())}
fig,ax=small(2.2,0.95); ax.barh([1,0],[base,oh],color=[GREEN,BLUE],height=0.6,edgecolor=NAVY,lw=0.5)
for yy,v in ((1,base),(0,oh)): ax.text(v+1,yy,'%.1f'%v,va='center',fontsize=7.8,color=NAVY)
ax.set_yticks([1,0]); ax.set_yticklabels(['target enc.','one-hot']); ax.set_xlim(0,48); ax.set_xticks([0,20,40]); ax.set_xlabel('5-fold MAE'); grid(ax); sv(fig,'p_59B1.png')

# 59C.1 readmissions: patients and years (schematic)
rng=np.random.RandomState(3); fig,ax=small(2.2,1.2)
for p_ in range(6):
    yrs=np.sort(rng.choice(np.arange(0,6),size=rng.randint(2,5),replace=False))
    ax.plot([0,5.4],[p_,p_],color=GRID,lw=0.6,zorder=1); ax.scatter(yrs,[p_]*len(yrs),s=14,color=BLUE,zorder=3)
ax.axvline(4.5,color=RED,lw=1.1,ls='--'); ax.text(4.62,5.6,'future',fontsize=7.8,color=RED,va='center',ha='left'); ax.text(4.38,5.6,'past',fontsize=7.8,color=NAVY,va='center',ha='right')
ax.set_ylim(-0.6,6.1); ax.set_xlim(-0.3,5.9); ax.set_yticks([0,5]); ax.set_yticklabels(['patient 1','6'],fontsize=7.8); ax.set_xticks([0,2,4]); ax.set_xlabel('year of stay'); sv(fig,'p_59C1.png')

# 59C.2 one column that explains everything
from sklearn.metrics import roc_auc_score
r_=np.random.RandomState(0); N=800; ycl=(r_.rand(N)<0.2).astype(int)
cols={'visits':r_.normal(0,1,N)+0.5*ycl,'tenure':r_.normal(0,1,N)-0.3*ycl,'spend':r_.normal(0,1,N)+0.2*ycl,'status':ycl*1.0+r_.normal(0,0.05,N)}
aucs={k:max(roc_auc_score(ycl,v),1-roc_auc_score(ycl,v)) for k,v in cols.items()}
A['59C.2']={'single_column_AUC':{k:round(v,3) for k,v in aucs.items()}}
fig,ax=small(2.2,1.0); ks=list(aucs)[::-1]; ax.barh(range(4),[aucs[k] for k in ks],color=[RED if k=='status' else BLUE for k in ks],height=0.6,edgecolor=NAVY,lw=0.5)
ax.set_yticks(range(4)); ax.set_yticklabels(ks); ax.set_xlim(0.4,1.05); ax.set_xticks([0.5,0.75,1.0]); ax.set_xlabel('AUC of one column'); grid(ax); sv(fig,'p_59C2.png')

# 59C.3 segments: no elbow
Xb,_=make_blobs(600,centers=5,cluster_std=2.2,random_state=0,center_box=(-8,8)); Kk=list(range(1,9)); inert=[KMeans(k,n_init=5,random_state=0).fit(Xb).inertia_ for k in Kk]
A['59C.3']={'inertia':[round(v) for v in inert]}
fig,ax=small(2.2,1.05); ax.plot(Kk,np.array(inert)/inert[0],color=GREEN,marker='o',ms=3.5,lw=1.3); ax.set_xlabel('K'); ax.set_ylabel('inertia (share)'); ax.set_xticks([1,4,8]); ax.set_yticks([0,0.5,1]); grid(ax); sv(fig,'p_59C3.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1,default=float); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print(json.dumps(A,indent=1,default=float))

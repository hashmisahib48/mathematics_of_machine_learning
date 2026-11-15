# Problem figures of Chapter 55. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import (KFold, StratifiedKFold, TimeSeriesSplit, GroupKFold, RepeatedStratifiedKFold,
    PredefinedSplit, cross_val_score, cross_val_predict, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
A={}; CHK={}
def small(w=1.8,h=1.0): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*115,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 55A.1 stratified allocation
y=np.array([0]*9+[1]*5); X=np.zeros((14,1)); cnt=[np.bincount(y[te],minlength=2).tolist() for _,te in StratifiedKFold(4).split(X,y)]
A['55A.1']={'class_counts_per_fold_[n0,n1]':cnt,'fold_sizes':[sum(c) for c in cnt]}
fig,ax=small(1.8,0.94); c=np.array(cnt); ax.bar(range(4),c[:,0],color=BLUE,width=0.6,label='class 0'); ax.bar(range(4),c[:,1],bottom=c[:,0],color=OR,width=0.6,hatch='//',edgecolor=NAVY,lw=0.5,label='class 1')
ax.set_xticks(range(4)); ax.set_xticklabels(['1','2','3','4']); ax.set_xlabel('test fold'); ax.set_ylabel('samples'); ax.set_ylim(0,7.2); ax.set_yticks([0,2,4]); ax.legend(ncol=2,loc='upper center',fontsize=7.8,handlelength=1,columnspacing=0.6,borderaxespad=0.0); sv(fig,'p_55A1.png')

# 55A.2 GroupKFold loads
sz=np.array([7,5,4,3,3,2,1]); g=np.repeat(np.arange(7),sz); fl=[]
for _,te in GroupKFold(3).split(np.zeros((len(g),1)),groups=g): fl.append((sorted(set(g[te].tolist())),len(te)))
A['55A.2']={'folds_[groups_0-based, rows]':fl}
fig,ax=small(1.8,0.94); ax.bar(range(3),[f[1] for f in fl],color=GREEN,width=0.6); ax.set_xticks(range(3)); ax.set_xticklabels(['1','2','3']); ax.set_xlabel('test fold'); ax.set_ylabel('rows'); ax.set_ylim(0,12.5); ax.set_yticks([0,5,10]); grid(ax)
for k,f in enumerate(fl): ax.text(k,f[1]+0.4,str(f[1]),ha='center',fontsize=7.8,color=NAVY)
sv(fig,'p_55A2.png')

# 55B.1 repeated stratified k-fold
Xb,yb=load_breast_cancer(return_X_y=True); pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
sc=cross_val_score(pipe,Xb,yb,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=0)); rm=sc.reshape(10,5).mean(1)
A['55B.1']={'mean':round(float(sc.mean()),4),'sd_50_scores':round(float(sc.std(ddof=1)),4),'sd_10_repeat_means':round(float(rm.std(ddof=1)),4),'min_repeat_mean':round(float(rm.min()),4),'max_repeat_mean':round(float(rm.max()),4)}
fig,ax=small(1.8,0.94); ax.plot(range(1,11),rm,'-o',color=BLUE,ms=3.5,lw=1.1); ax.axhline(sc.mean(),color=GREEN,lw=1.2); ax.set_xlabel('repeat'); ax.set_ylabel('5-fold mean'); ax.set_xticks([1,5,10]); grid(ax); sv(fig,'p_55B1.png')

# 55B.2 PredefinedSplit equals the hold-out
Xa,Xt,ya,yt=train_test_split(Xb,yb,test_size=0.25,random_state=0,stratify=yb)
Xall=np.r_[Xa,Xt]; yall=np.r_[ya,yt]; tf=np.r_[-np.ones(len(ya),int),np.zeros(len(yt),int)]
s_ps=cross_val_score(pipe,Xall,yall,cv=PredefinedSplit(tf)); s_ho=pipe.fit(Xa,ya).score(Xt,yt)
A['55B.2']={'n_splits':int(PredefinedSplit(tf).get_n_splits()),'predefined_score':round(float(s_ps[0]),4),'holdout_score':round(float(s_ho),4),'train_rows':len(ya),'test_rows':len(yt)}
fig,ax=small(1.8,0.68); ax.add_patch(Rectangle((0,0),426,1,fc=BLUE)); ax.add_patch(Rectangle((426,0),143,1,fc=OR)); ax.text(213,0.5,'-1: always train',ha='center',va='center',color='white',fontsize=7.8); ax.text(497,0.5,'0',ha='center',va='center',color=NAVY,fontsize=7.8)
ax.set_xlim(0,569); ax.set_ylim(0,1); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xticks([0,426,569]); ax.set_xlabel('row'); sv(fig,'p_55B2.png')

# 55C.1 stratification does not repair grouped leakage
cg=json.load(open('verify/case_groups.json'))
fig,ax=small(1.8,0.98); ks=['KFold, shuffled','StratifiedKFold','GroupKFold']; v=[cg[k][0] for k in ks]
ax.bar(range(3),v,color=[OR,OR,GREEN],width=0.6,hatch=None); ax.axhline(cg['truth'][0],color=PUR,lw=1.0,ls=':'); ax.set_xticks(range(3)); ax.set_xticklabels(['KF','Strat','Group'],fontsize=7.8); ax.set_ylim(0.4,1.0); ax.set_ylabel('accuracy'); grid(ax); sv(fig,'p_55C1.png')

# 55C.2 the gap in a forecasting split
fig,ax=small(1.8,0.94); rng=np.random.RandomState(2); e=np.zeros(60)
for k in range(1,60): e[k]=0.9*e[k-1]+rng.normal(0,0.5)
ax.plot(e,color=BLUE,lw=1.0); ax.axvspan(40,45,color=LGREY); ax.axvspan(45,50,color=OR,alpha=0.6); ax.text(36,1.2,'gap',ha='center',fontsize=7.8,color=NAVY); ax.text(54.5,1.5,'test',ha='center',fontsize=7.8,color=NAVY); ax.set_ylim(-2.5,2.4); ax.set_xlabel('week'); ax.set_ylabel('residual'); grid(ax); sv(fig,'p_55C2.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1); print(json.dumps(A)); print(CHK)

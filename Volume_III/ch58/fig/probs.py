# Problem figures of Chapter 58. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,itertools,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier as GBC, RandomForestClassifier as RF
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, make_scorer
from sklearn.model_selection import cross_val_predict, TunedThresholdClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bcdata import load
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.15): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*105,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]
Xa,Xb,ya,yb,names=load(); names=[str(n) for n in names]; ym,yt=1-ya,1-yb
lr=make_pipeline(StandardScaler(),LogisticRegression())

# 58A.1 threshold from costs
cfn,cfp=19,1; ts=cfp/(cfp+cfn)
A['58A.1']={'t_star':round(ts,4),'log_odds':round(float(np.log(cfp/cfn)),4),'say_yes_p004':bool(0.04*cfn>0.96*cfp),'say_yes_p006':bool(0.06*cfn>0.94*cfp),
  'cost_yes_004':round(0.96*cfp,3),'cost_no_004':round(0.04*cfn,3),'cost_yes_006':round(0.94*cfp,3),'cost_no_006':round(0.06*cfn,3)}
fig,ax=small(2.0,1.0); p=np.linspace(0,0.2,100); ax.plot(p,1-p,color=BLUE,lw=1.5); ax.plot(p,cfn*p,color=RED,lw=1.5); ax.plot([ts],[1-ts],'o',color=GREEN,ms=4.5)
ax.set_xlim(0,0.2); ax.set_ylim(0,1.2); ax.set_xlabel('$p$'); ax.set_ylabel('cost'); ax.set_xticks([0,0.05,0.2]); ax.set_yticks([0,0.5,1]); grid(ax); sv(fig,'p_58A1.png')

# 58A.2 expected accuracy after shuffling, enumerated
y=np.array([1,1,1,0,0,0,0,0]); yh=np.array([1,1,1,1,1,0,0,0]); base=(y==yh).mean()
accs=np.array([(yh[list(q)]==y).mean() for q in itertools.permutations(range(8))]); n1,m1=y.sum(),yh.sum(); form=(n1*m1+(8-n1)*(8-m1))/64
assert abs(accs.mean()-form)<1e-12
A['58A.2']={'baseline':base,'enumerated_mean_over_40320':round(float(accs.mean()),4),'formula':round(form,4),'importance':round(base-form,4)}
fig,ax=small(1.9,0.85); ax.hist(accs,bins=np.arange(-0.0625,1.0,0.125),color=LGREY,edgecolor=GREY,lw=0.6); ax.axvline(base,color=GREEN,lw=1.6); ax.axvline(form,color=RED,lw=1.6,ls='--')
ax.set_xlabel('accuracy after a shuffle'); ax.set_ylabel('orders'); ax.set_yticks([]); ax.set_xticks([0,0.5,1]); ax.spines['left'].set_visible(False); sv(fig,'p_58A2.png')

# 58A.3 partial dependence and ICE by hand
X=np.array([[1,2],[2,-1],[3,0],[4,1.]]); f=lambda v,x2:2*v+v*x2
pd_=lambda v:float(np.mean([f(v,x2) for x2 in X[:,1]]))
A['58A.3']={'PD_1':pd_(1),'PD_2':pd_(2),'PD_4':pd_(4),'ICE_slopes':[float(2+x2) for x2 in X[:,1]],'mean_x2':float(X[:,1].mean())}
fig,ax=small(2.0,1.15); v=np.array([1,4.])
for x2 in X[:,1]: ax.plot(v,f(v,x2),color=GREY,lw=0.9)
ax.plot(v,[pd_(1),pd_(4)],color=GREEN,lw=2.0); ax.set_xlabel('$v$'); ax.set_ylabel('$f$'); ax.set_xticks([1,2,3,4]); ax.set_yticks([0,8,16]); grid(ax); sv(fig,'p_58A3.png')

# 58B.1 recall targets on pooled CV scores
cv=cross_val_predict(lr,Xa,ya,cv=5,method='predict_proba')[:,0]; yp=(ya==0)
fpr,tpr,thr=roc_curve(yp,cv); out={}
for tg in (0.95,0.98,0.99):
    k=np.flatnonzero(tpr>=tg)[0]; yes=cv>=thr[k]; out[str(tg)]={'threshold':round(float(thr[k]),4),'recall':round(float(tpr[k]),4),'false_alarms':int((yes&~yp).sum()),'missed':int((~yes&yp).sum())}
A['58B.1']=out
fig,ax=small(2.0,1.2); ax.plot(fpr,tpr,color=BLUE,lw=1.4,drawstyle='steps-post')
for tg,mk in zip((0.95,0.98,0.99),('o','s','^')): k=np.flatnonzero(tpr>=tg)[0]; ax.plot([fpr[k]],[tpr[k]],mk,color=GREEN,ms=4.5)
ax.set_xlabel('false alarm rate'); ax.set_ylabel('recall'); ax.set_xlim(0,0.3); ax.set_ylim(0.9,1.005); ax.set_xticks([0,0.1,0.2,0.3]); ax.set_yticks([0.9,1.0]); grid(ax); sv(fig,'p_58B1.png')

# 58B.2 the spread of a permutation importance against n_repeats
gb=GBC(random_state=0).fit(Xa,ym); jj=names.index('worst texture'); out={}
for R in (1,4,16):
    v=[permutation_importance(gb,Xb,yt,scoring='neg_log_loss',n_repeats=R,random_state=s).importances_mean[jj] for s in range(10)]; out[R]={'mean':round(float(np.mean(v)),4),'sd':round(float(np.std(v)),4)}
A['58B.2']=out
fig,ax=small(2.0,1.15); sds=[out[R]['sd'] for R in (1,4,16)]; ax.plot([1,4,16],sds,'o-',color=PUR,lw=1.3,ms=4.5); ax.plot([1,4,16],[sds[0]/np.sqrt(r) for r in (1,4,16)],':',color=GREY,lw=1.2)
ax.set_xscale('log'); ax.set_xticks([1,4,16]); ax.set_xticklabels(['1','4','16']); ax.set_xlabel('n_repeats'); ax.set_ylabel('sd'); ax.set_ylim(0,0.04); ax.set_yticks([0,0.02,0.04]); grid(ax); sv(fig,'p_58B2.png')

# 58C.1 scores of a forest are not probabilities
rf=RF(n_estimators=200,random_state=0).fit(Xa,ya); pr=rf.predict_proba(Xb)[:,0]; pl=lr.fit(Xa,ya).predict_proba(Xb)[:,0]
fr,mp=calibration_curve(yb==0,pr,n_bins=5,strategy='uniform'); fl,ml=calibration_curve(yb==0,pl,n_bins=5,strategy='uniform')
A['58C.1']={'forest_bins':{'observed':r4(fr),'predicted':r4(mp)},'logistic_bins':{'observed':r4(fl),'predicted':r4(ml)},'forest_scores_below_0.05':int((pr<0.05).sum()),'forest_scores_above_0.95':int((pr>0.95).sum()),'forest_min_nonzero':round(float(pr[pr>0].min()),4)}
fig,ax=small(2.0,1.3); ax.plot([0,1],[0,1],color=GREY,lw=0.8,ls=':'); ax.plot(mp,fr,'o-',color=PUR,lw=1.2,ms=4); ax.plot(ml,fl,'s--',color=BLUE,lw=1.2,ms=4)
ax.set_xlabel('forest score $p$'); ax.set_ylabel('share malignant'); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); grid(ax); sv(fig,'p_58C1.png')

# 58C.2 a flat partial dependence hides an interaction
rg=np.random.RandomState(3); Xi=rg.uniform(-1,1,(600,2)); yi=((Xi[:,0]*Xi[:,1]+0.1*rg.standard_normal(600))>0).astype(int)
gi=GBC(random_state=0).fit(Xi,yi)
pdi=partial_dependence(gi,Xi,[0],kind='both',grid_resolution=40,method='brute',response_method='predict_proba')
g1=pdi['grid_values'][0]; ice=pdi['individual'][0]; avg=pdi['average'][0]
A['58C.2']={'train_acc':round(gi.score(Xi,yi),4),'pd_min':round(float(avg.min()),4),'pd_max':round(float(avg.max()),4),
  'ice_pos_x2_mean_first_last':[round(float(ice[Xi[:,1]>0].mean(0)[0]),3),round(float(ice[Xi[:,1]>0].mean(0)[-1]),3)],'ice_neg_x2_mean_first_last':[round(float(ice[Xi[:,1]<0].mean(0)[0]),3),round(float(ice[Xi[:,1]<0].mean(0)[-1]),3)]}
fig,ax=small(2.0,1.3); idx=np.random.RandomState(0).choice(600,40,replace=False)
for i_ in idx: ax.plot(g1,ice[i_],color=OR if Xi[i_,1]>0 else BLUE,lw=0.5,alpha=0.6)
ax.plot(g1,avg,color=GREEN,lw=2.2); ax.set_xlabel('$x_1$'); ax.set_ylabel('$P$(class 1)'); ax.set_ylim(-0.03,1.03); ax.set_yticks([0,0.5,1]); ax.set_xticks([-1,0,1]); grid(ax); sv(fig,'p_58C2.png')

# the cut chosen on the test rows (quoted in the warning of Section 58.3)
def cost(y,pred,cfn,cfp,pos_label):
    miss=(y==pos_label)&(pred!=pos_label); alarm=(y!=pos_label)&(pred==pos_label); return (cfn*miss.sum()+cfp*alarm.sum())/len(y)
c10=make_scorer(cost,greater_is_better=False,cfn=10,cfp=1,pos_label=0)
tc=TunedThresholdClassifierCV(lr,scoring=c10,cv=5).fit(Xa,ya); pt=tc.estimator_.predict_proba(Xb)[:,0]
cv_cost=cost(yb,np.where(pt>=tc.best_threshold_,0,1),10,1,0); ts_cost=cost(yb,np.where(pt>=1/11,0,1),10,1,0)
grid_t=np.unique(np.r_[0,pt,1.01]); hind=[cost(yb,np.where(pt>=t,0,1),10,1,0) for t in grid_t]; kb=int(np.argmin(hind))
A['test_cut_warning']={'cv_threshold':round(float(tc.best_threshold_),4),'cv_threshold_test_cost':round(cv_cost,4),'t_star_test_cost':round(ts_cost,4),'hindsight_threshold':round(float(grid_t[kb]),4),'hindsight_test_cost':round(float(hind[kb]),4),
  'n_thresholds_giving_min':int(np.sum(np.isclose(hind,hind[kb])))}

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

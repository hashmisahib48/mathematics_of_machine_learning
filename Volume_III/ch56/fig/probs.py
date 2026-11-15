# Problem figures of Chapter 56. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,math,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import loguniform, uniform
from sklearn.datasets import load_diabetes, load_breast_cancer, make_classification
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, ParameterGrid, train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import cancer_split
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
A={}; CHK={}
def small(w=1.8,h=1.0): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*100,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# 56A.1 a halving schedule (100 candidates, factor 3, 1800 rows)
n0,eta,R=100,3,1800; req=1+int(math.floor(math.log(n0,eta))); r0=max(20,R//eta**(req-1)); poss=1+int(math.floor(math.log(R//r0,eta))); nr=min(req,poss)
nk=[math.ceil(n0/eta**k) for k in range(nr)]; rk=[min(eta**k*r0,R) for k in range(nr)]
Xh,yh=make_classification(n_samples=R,n_informative=6,random_state=0)
g100={'max_depth':list(range(1,11)),'min_samples_leaf':[1,2,3,5,8,12,18,25,35,50]}
hs=HalvingGridSearchCV(DecisionTreeClassifier(random_state=0),g100,factor=eta,cv=StratifiedKFold(5,shuffle=True,random_state=0),random_state=0).fit(Xh,yh)
assert hs.n_candidates_==nk and hs.n_resources_==rk
A['56A.1']={'n_required':req,'n_possible':poss,'r0':r0,'candidates':nk,'rows':rk,'total_rows':int(sum(a*b for a,b in zip(nk,rk))),'full_grid_rows':n0*R,'sklearn_candidates':hs.n_candidates_,'sklearn_rows':hs.n_resources_}
fig,ax=small(1.5,0.8); ax.bar(range(nr),[a*b for a,b in zip(nk,rk)],color=PUR,width=0.6,edgecolor=NAVY,lw=0.5); ax.set_xticks(range(nr)); ax.set_xlabel('rung $k$'); ax.set_ylabel('$n_k r_k$'); ax.set_ylim(0,4600); ax.set_yticks([0,2000,4000]); grid(ax); sv(fig,'p_56A1.png')

# 56A.2 random draws needed
pr={}
for q in (0.02,0.05,0.10): pr[q]=math.ceil(math.log(0.01)/math.log(1-q))
A['56A.2']={'n_for_P0.99_q0.02':pr[0.02],'P_with_60_draws_q0.02':round(1-0.98**60,4),'n_for_P0.99_q0.05':pr[0.05],'n_for_P0.99_q0.10':pr[0.10],'check_P_at_228':round(1-0.98**228,4)}
fig,ax=small(1.5,0.8); nn=np.arange(0,301)
for q,col,ls in ((0.02,PUR,'-'),(0.05,BLUE,'--'),(0.10,GREEN,':')): ax.plot(nn,1-(1-q)**nn,color=col,ls=ls,lw=1.3,label='$q$ = %.2f'%q)
ax.axhline(0.99,color=GREY,lw=0.7); ax.set_xlabel('draws $n$'); ax.set_ylabel('$P$'); ax.set_ylim(0,1.05); ax.set_xticks([0,100,200,300]); ax.legend(loc='lower right',fontsize=7.2,handlelength=1.5,labelspacing=0.1,borderaxespad=0.1); grid(ax); sv(fig,'p_56A2.png')

# 56B.1 the one-standard-error rule as a refit callable
Xb,Xbt,yb,ybt=cancer_split()
def one_se(res):
    m=res['mean_test_score']; s=res['std_test_score']; b=int(np.argmax(m)); thr=m[b]-s[b]/np.sqrt(5)
    ok=np.flatnonzero(m>=thr); Cs=np.array([p['logisticregression__C'] for p in res['params']]); return int(ok[np.argmin(Cs[ok])])
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)); Cg=np.logspace(-4,2,13)
g1=GridSearchCV(lr,{'logisticregression__C':Cg},cv=5).fit(Xb,yb); g2=GridSearchCV(lr,{'logisticregression__C':Cg},cv=5,refit=one_se).fit(Xb,yb)
A['56B.1']={'argmax_C':float(g1.best_params_['logisticregression__C']),'argmax_cv':round(float(g1.best_score_),4),'argmax_test':round(float(g1.score(Xbt,ybt)),4),'one_se_C':float(g2.best_params_['logisticregression__C']),'one_se_cv':round(float(g2.cv_results_['mean_test_score'][g2.best_index_]),4),'one_se_test':round(float(g2.score(Xbt,ybt)),4),'best_score_available_with_callable':hasattr(g2,'best_score_')}
r=g1.cv_results_; fig,ax=small(1.5,0.84); ax.errorbar(np.log10(Cg),r['mean_test_score'],yerr=r['std_test_score']/np.sqrt(5),color=BLUE,lw=1.0,marker='o',ms=2.8,capsize=1.5)
ax.axvline(np.log10(A['56B.1']['one_se_C']),color=GREEN,lw=1.2); ax.axvline(np.log10(A['56B.1']['argmax_C']),color=OR,lw=1.2,ls='--'); ax.set_xlabel('log$_{10}\\,C$'); ax.set_ylabel('accuracy'); ax.set_ylim(0.93,1.0); grid(ax); sv(fig,'p_56B1.png')

# 56B.2 halving with the number of trees as the resource
rg={'max_depth':[2,4,8,None],'max_features':['sqrt',0.5,1.0],'min_samples_leaf':[1,3,9]}
hr=HalvingGridSearchCV(RandomForestClassifier(random_state=0),rg,resource='n_estimators',max_resources=81,min_resources=3,factor=3,cv=5,random_state=0).fit(Xb,yb)
A['56B.2']={'candidates':hr.n_candidates_,'n_estimators':hr.n_resources_,'trees_fitted_per_fold':int(sum(a*b for a,b in zip(hr.n_candidates_,hr.n_resources_))),'full_grid_trees_per_fold':36*81,'best_score':round(float(hr.best_score_),4),'best_params':{k:(v if isinstance(v,(int,str,type(None))) else float(v)) for k,v in hr.best_params_.items()},'test':round(float(hr.score(Xbt,ybt)),4),'best_estimator_n_estimators':int(hr.best_estimator_.n_estimators)}
fig,ax=small(1.5,0.8); ax.bar(range(len(hr.n_candidates_)),hr.n_candidates_,color=BLUE,width=0.6,edgecolor=NAVY,lw=0.5); ax.set_xticks(range(len(hr.n_candidates_))); ax.set_xticklabels([str(x) for x in hr.n_resources_]); ax.set_xlabel('trees $r_k$'); ax.set_ylabel('$n_k$'); ax.set_ylim(0,42); grid(ax); sv(fig,'p_56B2.png')

# 56C.1 the gaps of best_score_ and of the nested score
W=np.array(json.load(open('verify/winner.json'))); A['56C.1']=None
fig,ax=small(1.5,0.8); m_=W.mean(axis=0); ax.bar(range(3),m_,color=[OR,GREEN,'#B8BEC8'],width=0.6,edgecolor=NAVY,lw=0.5); ax.set_ylim(0.96,0.98); ax.set_yticks([0.96,0.97,0.98]); ax.set_xticks(range(3)); ax.set_xticklabels(['inner','nested','fresh'],fontsize=7.8); ax.set_ylabel('mean accuracy'); grid(ax); sv(fig,'p_56C1.png')

# 56C.2 uniform against log-uniform draws
rs_=np.random.RandomState(0); u=uniform(1e-4,100-1e-4).rvs(20000,random_state=rs_); l=loguniform(1e-4,100).rvs(20000,random_state=rs_)
fr=lambda x:[round(float(((x>=10.0**a)&(x<10.0**(a+1))).mean()),3) for a in range(-4,2)]
A['56C.2']={'decade_shares_uniform_1e-4..1e2':fr(u),'decade_shares_loguniform':fr(l)}
fig,ax=small(1.5,0.8); w=0.36; ax.bar(np.arange(6)-w/2,fr(u),w,color=OR,hatch='//',edgecolor=NAVY,lw=0.4); ax.bar(np.arange(6)+w/2,fr(l),w,color=BLUE,edgecolor=NAVY,lw=0.4)
ax.set_xticks(range(6)); ax.set_xticklabels(['-4','-3','-2','-1','0','1']); ax.set_xlabel('decade, log$_{10}\\,x$'); ax.set_ylabel('share'); ax.set_ylim(0,1.0); grid(ax); sv(fig,'p_56C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/problem_textsize.json','w'),indent=1)
print(json.dumps(A,indent=1)); print(CHK)

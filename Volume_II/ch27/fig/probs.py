# Problem figures of Chapter 27. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.special import expit
from sklearn.ensemble import (GradientBoostingRegressor as GR, GradientBoostingClassifier as GC, AdaBoostClassifier)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.datasets import load_diabetes, load_breast_cancer, make_friedman1
from sklearn.model_selection import train_test_split
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=1.95,h=1.0): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*104,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def inv_cdf(e,q): return float(np.percentile(e,q,method='inverted_cdf'))

# 27A.1 two rounds with eta = 1
x1=np.array([1,2,3,4.]); y1=np.array([2,3,9,12.]); g=GR(n_estimators=2,max_depth=1,learning_rate=1.0,random_state=0).fit(x1[:,None],y1)
st=list(g.staged_predict(x1[:,None]))
A['27A.1']={'F0':float(g.init_.constant_.item()),'r1':[float(v) for v in y1-g.init_.constant_.item()],'thr':[float(e[0].tree_.threshold[0]) for e in g.estimators_],
  'leaves':[[round(float(v),4) for v in e[0].tree_.value[1:3,0,0]] for e in g.estimators_],'F1':[float(v) for v in st[0]],'F2':[float(v) for v in st[1]],
  'r2':[float(v) for v in y1-st[0]],'mse':[round(float(v),4) for v in g.train_score_],'mse0':round(float(np.mean((y1-6.5)**2)),4)}
fig,ax=small(); ax.scatter(x1,y1,s=36,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xticks(x1); ax.set_ylim(0,14); grid(ax); sv(fig,'p_27A1.png')

# 27A.2 absolute error against squared error with an outlier
x2=np.arange(1,7.); y2=np.array([1,2,4,5,6,50.]); med0=float(np.median(y2)); r_abs=y2-med0; left=x2<=3.5
ga=GR(loss='absolute_error',n_estimators=1,max_depth=1,learning_rate=1.0,random_state=0).fit(x2[:,None],y2)
gs_=GR(n_estimators=1,max_depth=1,learning_rate=1.0,random_state=0).fit(x2[:,None],y2)
A['27A.2']={'median':med0,'resid':[float(v) for v in r_abs],'abs_thr':float(ga.estimators_[0,0].tree_.threshold[0]),'abs_leaves':[inv_cdf(r_abs[left],50),inv_cdf(r_abs[~left],50)],
  'abs_pred':[float(v) for v in ga.predict(x2[:,None])],'mean0':float(y2.mean()),'sq_thr':float(gs_.estimators_[0,0].tree_.threshold[0]),
  'sq_leaves':[round(float(v),4) for v in gs_.estimators_[0,0].tree_.value[1:3,0,0]],'sq_pred':[round(float(v),4) for v in gs_.predict(x2[:,None])]}
fig,ax=small(); ax.scatter(x2,y2,s=34,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xticks(x2); ax.set_ylim(-3,55); grid(ax); sv(fig,'p_27A2.png')

# 27A.3 a Newton leaf
yl=np.array([1,1,0.]); pl=np.array([0.2,0.5,0.8]); rl=yl-pl; gam=float(rl.sum()/(pl*(1-pl)).sum()); F=np.log(pl[0]/(1-pl[0]))
A['27A.3']={'r':[round(float(v),4) for v in rl],'num':round(float(rl.sum()),4),'den':round(float((pl*(1-pl)).sum()),4),'gamma':round(gam,4),'F_old':round(float(F),4),'F_new':round(float(F+0.5*gam),4),'p_new':round(float(expit(F+0.5*gam)),4)}
fig,ax=small(); Fv=np.linspace(-4,4,200); ax.plot(Fv,expit(Fv),color=GREY,lw=1.2)
for k,(pp,yy) in enumerate(zip(pl,yl)): ax.scatter(np.log(pp/(1-pp)),pp,s=40,marker='o' if yy==0 else 's',color=BLUE if yy==0 else OR,edgecolor='white',lw=0.7,zorder=5)
ax.set_xlabel('score $F$'); ax.set_ylabel('$p=\\sigma(F)$'); grid(ax); sv(fig,'p_27A3.png')

# 27A.4 three classes, one leaf
yk=np.array([0,0,1,1,2,2]); K=3; P=np.full((6,K),1/3)
leaf=np.array([True,True,True,False,False,False]); res={}
for k in range(K):
    r=(yk==k)-P[:,k]; num=r[leaf].sum(); den=(P[:,k]*(1-P[:,k]))[leaf].sum(); res[k]={'sum_r':round(float(num),4),'sum_pq':round(float(den),4),'gamma_plain':round(float(num/den),4),'gamma_sk':round(float((K-1)/K*num/den),4)}
A['27A.4']=res
fig,ax=small(2.4,1.5)
for i_,k_ in enumerate(yk): ax.scatter(i_+1,0,s=44,marker=['o','s','^'][k_],color=[BLUE,OR,GREEN][k_],edgecolor='white',lw=0.7,zorder=5)
ax.axvline(3.5,color=GREEN,lw=1.2,ls='--'); ax.set_yticks([]); ax.set_xticks(range(1,7)); ax.set_xlabel('sample $i$ (marker = class)'); ax.set_ylim(-0.6,0.6); ax.spines['left'].set_visible(False); sv(fig,'p_27A4.png')

# 27B.1 absolute-error booster from scratch
Xd,yd=load_diabetes(return_X_y=True); Xdt,Xde,ydt,yde=train_test_split(Xd,yd,test_size=0.25,random_state=0)
class MyAbs:
    def __init__(s,n=100,lr=0.1,d=2): s.n,s.lr,s.d=n,lr,d
    def fit(s,X,y):
        rng=check_random_state(0); s.f0=np.median(y); F=np.full(len(y),s.f0); s.trees=[]; s.leaf=[]
        for m in range(s.n):
            r=y-F; t=DecisionTreeRegressor(max_depth=s.d,random_state=rng).fit(X,np.sign(r)); lf=t.apply(X); gg=np.zeros(t.tree_.node_count)
            for l in np.unique(lf): gg[l]=inv_cdf(r[lf==l],50)
            F+=s.lr*gg[lf]; s.trees.append(t); s.leaf.append(gg)
        return s
    def predict(s,X):
        F=np.full(len(X),s.f0)
        for t,gg in zip(s.trees,s.leaf): F+=s.lr*gg[t.apply(X)]
        return F
a_=MyAbs().fit(Xdt,ydt); b_=GR(loss='absolute_error',n_estimators=100,max_depth=2,random_state=0).fit(Xdt,ydt)
A['27B.1']={'max_diff':float(np.abs(a_.predict(Xde)-b_.predict(Xde)).max()),'mae_test':round(float(np.abs(yde-b_.predict(Xde)).mean()),3),'r2_test':round(float(b_.score(Xde,yde)),4)}
fig,ax=small(); e=np.linspace(-3,3,300); ax.plot(e,np.sign(e),color=GREEN,lw=1.6); ax.axhline(0,color=GREY,lw=0.6); ax.set_xlabel('residual $e$'); ax.set_ylabel('pseudo-residual'); ax.set_ylim(-1.4,1.4); grid(ax); sv(fig,'p_27B1.png')

# 27B.2 exponential loss against AdaBoost
Xc,yc=load_breast_cancer(return_X_y=True); Xct,Xce,yct,yce=train_test_split(Xc,yc,test_size=0.25,random_state=0,stratify=yc)
ge=GC(loss='exponential',n_estimators=100,max_depth=1,random_state=0).fit(Xct,yct); ad=AdaBoostClassifier(n_estimators=100,random_state=0).fit(Xct,yct)
gl=GC(n_estimators=100,max_depth=1,random_state=0).fit(Xct,yct)
A['27B.2']={'gb_exp':round(float(ge.score(Xce,yce)),4),'ada':round(float(ad.score(Xce,yce)),4),'gb_log':round(float(gl.score(Xce,yce)),4),
  'agree_exp_ada':round(float((ge.predict(Xce)==ad.predict(Xce)).mean()),4),'p_formula':bool(np.allclose(ge.predict_proba(Xce)[:,1],expit(2*ge.decision_function(Xce))))}
fig,ax=small(); mg=np.linspace(-2,2,300); ax.plot(mg,np.exp(-mg),color=RED,lw=1.6,label='$e^{-m}$'); ax.plot(mg,np.log1p(np.exp(-2*mg))/np.log(2),color=BLUE,lw=1.6,ls='--',label='log loss')
ax.set_ylim(0,4); ax.set_xlabel('margin $m = sF$'); ax.legend(loc='upper right',fontsize=9,borderaxespad=0.1,handlelength=1.2,labelspacing=0.2); grid(ax); sv(fig,'p_27B2.png')

# 27B.3 the stage chosen by early stopping varies with the seed
Xf,yf=make_friedman1(n_samples=600,noise=1.0,random_state=0); Xft,Xfe,yft,yfe=train_test_split(Xf,yf,test_size=0.5,random_state=0)
stops=[]
for sd in range(10):
    g=GR(n_estimators=1000,learning_rate=0.1,n_iter_no_change=10,random_state=sd).fit(Xft,yft); stops.append((int(g.n_estimators_),round(float(g.score(Xfe,yfe)),4)))
A['27B.3']={'stops':stops,'min':min(s[0] for s in stops),'max':max(s[0] for s in stops),'r2_min':min(s[1] for s in stops),'r2_max':max(s[1] for s in stops)}
fig,ax=small(); ax.plot(range(10),[s[0] for s in stops],'o',color=BLUE,ms=5); ax.set_xlabel('random_state'); ax.set_ylabel('trees kept'); ax.set_xticks([0,3,6,9]); grid(ax); sv(fig,'p_27B3.png')

# 27C.1 stumps and the exclusive-or
rs=np.random.RandomState(0); Xx=rs.uniform(-1,1,(240,2)); yx=((Xx[:,0]>0)^(Xx[:,1]>0)).astype(int)
Xxt,Xxe,yxt,yxe=Xx[:160],Xx[160:],yx[:160],yx[160:]; res={}
for d in (1,2): m=GC(max_depth=d,n_estimators=100,random_state=0).fit(Xxt,yxt); res[d]={'train':round(float(m.score(Xxt,yxt)),4),'test':round(float(m.score(Xxe,yxe)),4)}
A['27C.1']=res
fig,ax=small();
for cl_,col,mk in ((0,BLUE,'o'),(1,OR,'s')): mm=yx==cl_; ax.scatter(Xx[mm,0],Xx[mm,1],s=7,color=col,marker=mk,edgecolor='none',zorder=5)
ax.axhline(0,color=GREEN,lw=0.9); ax.axvline(0,color=GREEN,lw=0.9); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); sv(fig,'p_27C1.png')

# 27C.2 an out-of-bag score that looks too good
g=GR(learning_rate=0.1,n_estimators=300,max_depth=3,subsample=0.5,random_state=0).fit(Xft,yft)
from sklearn.metrics import mean_squared_error
te=np.array([mean_squared_error(yfe,p) for p in g.staged_predict(Xfe)])
A['27C.2']={'oob_last':round(float(g.oob_scores_[-1]),4),'test_last':round(float(te[-1]),3),'train_last':round(float(g.train_score_[-1]),4),'oob_at_50':round(float(g.oob_scores_[49]),3),'test_at_50':round(float(te[49]),3)}
fig,ax=small(); ax.plot(range(1,301),te,color=GREY,lw=1.5,label='test'); ax.plot(range(1,301),g.oob_scores_,color=RED,lw=1.5,ls=':',label='out-of-bag'); ax.set_ylim(0,8); ax.set_xlabel('trees'); ax.set_ylabel('mean squared error'); ax.legend(loc='upper right',fontsize=9,borderaxespad=0.1,handlelength=1.8); grid(ax); sv(fig,'p_27C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

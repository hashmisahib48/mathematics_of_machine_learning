# Problem figures of Chapter 25. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import BaggingClassifier as BAG, RandomForestClassifier as RFC
from sklearn.datasets import load_breast_cancer, make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# 25A.1 bootstrap counts
n=8; draw=[2,5,5,1,7,2,2,8]; cnt=np.bincount(np.array(draw)-1,minlength=n)
A['25A.1']={'counts':[int(c) for c in cnt],'out_of_bag':[int(i+1) for i in range(n) if cnt[i]==0],'distinct':int((cnt>0).sum()),'p_out':round((1-1/n)**n,4),'expected_distinct':round(n*(1-(1-1/n)**n),3)}
fig,ax=small(); 
for i in range(n):
    if cnt[i]>0: ax.bar(i+1,cnt[i],color=BLUE,width=0.65)
    else: ax.bar(i+1,0.12,color='white',edgecolor=BLUE,lw=1.2,width=0.65,hatch='///')
ax.set_xticks(range(1,n+1)); ax.set_xlabel('point'); ax.set_ylabel('times drawn'); ax.set_yticks([0,1,2,3]); ax.set_ylim(0,3.4); grid(ax); sv(fig,'p_25A1.png')

# 25A.2 variance of an average
s2,rho=9.0,0.4; var=lambda B: rho*s2+(1-rho)*s2/B
A['25A.2']={'B10':round(var(10),4),'floor':rho*s2,'B_for_10pct':int(np.ceil((1-rho)/(0.1*rho))),'var_at_that_B':round(var(int(np.ceil((1-rho)/(0.1*rho)))),4),'ratio_at_that_B':round(var(15)/(rho*s2),4)}
Bg=np.linspace(1,40,200); fig,ax=small(); ax.plot(Bg,var(Bg),color=GREEN,lw=1.7); ax.axhline(rho*s2,color=PUR,lw=1.0,ls='--'); ax.text(39,rho*s2+0.25,'$\\rho\\sigma^2$',color=PUR,fontsize=9,ha='right',va='bottom')
ax.set_xlabel('$B$'); ax.set_ylabel('variance'); ax.set_ylim(0,9.6); ax.set_xlim(0,40); grid(ax); sv(fig,'p_25A2.png')

# 25A.3 soft and hard voting
P3=np.array([[0.9,0.4,0.45],[0.6,0.6,0.1]]); A['25A.3']={'mean':[round(float(v),4) for v in P3.mean(1)],'soft':[int(v>0.5) for v in P3.mean(1)],'votes':[[int(q>0.5) for q in r] for r in P3],'hard':[int((r>0.5).sum()*2>3) for r in P3]}
fig,ax=small()
for k,r in enumerate(P3):
    ax.scatter([k+0.82,k+1,k+1.18],r,s=34,color=[BLUE,GREEN,PUR],marker='o',zorder=5); ax.plot([k+0.65,k+1.35],[r.mean()]*2,color=RED,lw=1.4)
ax.axhline(0.5,color=GREY,lw=0.8,ls=':'); ax.set_xticks([1,2]); ax.set_xticklabels(['sample 1','sample 2']); ax.set_xlim(0.4,2.6); ax.set_ylim(0,1.05); ax.set_ylabel('$P$(class 1)'); grid(ax); sv(fig,'p_25A3.png')

# 25A.4 random thresholds
xv=np.array([2,3,7,9.]); yv=np.array([0,0,1,1]); g=lambda y: 1-((np.bincount(y,minlength=2)/len(y))**2).sum() if len(y) else 0
def dec(t): L=yv[xv<=t]; R=yv[xv>t]; return g(yv)-len(L)/4*g(L)-len(R)/4*g(R)
A['25A.4']={'root_gini':g(yv),'dec':{str(t):round(float(dec(t)),4) for t in (2.5,4.0,8.0,5.0)},'p_perfect':round(4/7,4)}
fig,ax=small(2.4,1.3)
for xx,yy in zip(xv,yv): ax.scatter(xx,0,s=45,marker='o' if yy==0 else 's',color=BLUE if yy==0 else OR,edgecolor='white',lw=0.7,zorder=5)
for t,ls in zip((2.5,4.0,8.0),('--','-.',':')): ax.axvline(t,color=PUR,lw=1.1,ls=ls); ax.text(t+(-0.12 if t==2.5 else 0.12),0.42,f'{t:g}',color=PUR,fontsize=9,ha='right' if t==2.5 else 'left',va='top')
ax.set_yticks([]); ax.set_xticks([2,3,7,9]); ax.set_xlim(1,10); ax.set_ylim(-0.5,0.5); ax.spines['left'].set_visible(False); ax.set_xlabel('$x$'); sv(fig,'p_25A4.png')

# 25B.1 manual OOB of BaggingClassifier
cd=load_breast_cancer(); Xt,Xe,yt,ye=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
bg=BAG(DTC(),n_estimators=50,oob_score=True,random_state=0).fit(Xt,yt)
Pm=np.zeros((len(yt),2)); 
for t,idx in zip(bg.estimators_,bg.estimators_samples_):
    out=np.ones(len(yt),bool); out[idx]=False; Pm[out]+=t.predict_proba(Xt[out])
man=float((Pm.argmax(1)==yt).mean()); A['25B.1']={'oob_score_':round(float(bg.oob_score_),4),'manual':round(man,4),'equal':bool(abs(man-bg.oob_score_)<1e-12),'n_no_oob':int((Pm.sum(1)==0).sum())}
fig,ax=small(); nin=[len(np.unique(i)) for i in bg.estimators_samples_[:50]]; ax.hist(nin,bins=np.arange(min(nin)-0.5,max(nin)+1.5,2),color=BLUE,edgecolor='white'); ax.axvline(0.632*len(yt),color=PUR,lw=1.1,ls='--'); ax.set_xlabel('distinct samples per tree'); ax.set_ylabel('trees'); grid(ax); sv(fig,'p_25B1.png')
A['25B.1']['mean_distinct']=round(float(np.mean(nin)),2); A['25B.1']['0.632n']=round(0.632*len(yt),1)

# 25B.2 bagging stable and unstable learners (moons)
Xm,ym=make_moons(n_samples=300,noise=0.3,random_state=0); Xa,Xb,ya,yb=train_test_split(Xm,ym,test_size=0.33,random_state=0,stratify=ym); res={}
for name,d in (('stump',1),('depth 3',3),('full',None)):
    single=DTC(max_depth=d,random_state=0).fit(Xa,ya); bagd=BAG(DTC(max_depth=d),n_estimators=100,random_state=0).fit(Xa,ya)
    res[name]={'single':round(float(single.score(Xb,yb)),4),'bagged':round(float(bagd.score(Xb,yb)),4)}
A['25B.2']=res
fig,ax=small(); xx=np.arange(3); ax.bar(xx-0.19,[res[k]['single'] for k in res],0.36,color=GREY,label='one tree'); ax.bar(xx+0.19,[res[k]['bagged'] for k in res],0.36,color=GREEN,label='bagged')
ax.set_xticks(xx); ax.set_xticklabels(['stump','depth 3','full']); ax.set_ylim(0.6,1.0); ax.set_ylabel('test accuracy'); ax.legend(loc='upper left',fontsize=9,borderaxespad=0.1); grid(ax); sv(fig,'p_25B2.png')

# 25B.3 class weights on an imbalanced problem
X,y=make_classification(n_samples=2000,n_features=10,n_informative=4,weights=[0.93,0.07],flip_y=0.02,class_sep=0.8,random_state=0)
Xa,Xb,ya,yb=train_test_split(X,y,test_size=0.5,random_state=0,stratify=y); r3={}
for cw in (None,'balanced','balanced_subsample'):
    f=RFC(200,class_weight=cw,min_samples_leaf=5,random_state=0).fit(Xa,ya); p=f.predict(Xb)
    r3[str(cw)]={'recall':round(float(recall_score(yb,p)),3),'precision':round(float(precision_score(yb,p)),3),'accuracy':round(float((p==yb).mean()),3)}
A['25B.3']={'results':r3,'minority_share_test':round(float(yb.mean()),3),'n_minority_test':int(yb.sum())}
fig,ax=small(); ks=list(r3); xx=np.arange(3); ax.bar(xx-0.19,[r3[k]['recall'] for k in ks],0.36,color=OR,label='recall'); ax.bar(xx+0.19,[r3[k]['precision'] for k in ks],0.36,color=BLUE,label='precision')
ax.set_xticks(xx); ax.set_xticklabels(['none','balanced','b.-sub.'],fontsize=8.6); ax.set_ylim(0,1.18); ax.legend(loc='upper left',fontsize=9,borderaxespad=0.1,ncol=2,columnspacing=0.8,handlelength=1.0); grid(ax); sv(fig,'p_25B3.png')

# 25C.1 more trees
rows=json.load(open('verify/fig_numbers.json'))['oobcurve']; A['25C.2']={'oob_mean_100plus':rows['oob_mean_100plus'],'test_mean_100plus':rows['test_mean_100plus']}
fig,ax=small()
f=RFC(n_estimators=300,random_state=0).fit(Xt,yt); Pte=np.cumsum([t.predict_proba(Xe) for t in f.estimators_],axis=0); err=[(Pte[b].argmax(1)!=ye).mean() for b in range(300)]
A['25C.2'].update({'err_B1':round(float(err[0]),4),'err_min':round(float(min(err)),4),'B_at_min':int(np.argmin(err))+1,'err_B300':round(float(err[-1]),4),'n_test':int(len(ye))})
ax.plot(np.arange(1,301),err,color=BLUE,lw=1.2); ax.set_xscale('log'); ax.set_xlabel('trees $B$'); ax.set_ylabel('test error'); ax.set_xticks([1,10,100]); ax.set_xticklabels(['1','10','100']); grid(ax); sv(fig,'p_25C1.png')

# 25C.2 a duplicated feature
rng=np.random.default_rng(0); n=1000; s_=rng.integers(0,3,n); y_=rng.random(n)<0.2+0.3*s_; u_=rng.uniform(size=n)
X1=np.c_[s_,u_]; X2=np.c_[s_,s_,u_]; f1=RFC(300,random_state=0).fit(X1[:500],y_[:500]); f2=RFC(300,random_state=0).fit(X2[:500],y_[:500])
A['25C.1']={'one_copy':[round(float(v),3) for v in f1.feature_importances_],'two_copies':[round(float(v),3) for v in f2.feature_importances_],'test_one':round(float(f1.score(X1[500:],y_[500:])),3),'test_two':round(float(f2.score(X2[500:],y_[500:])),3)}
fig,ax=small(); ax.bar([0,1,2],f2.feature_importances_,color=[GREEN,GREEN,GREY],width=0.62); ax.set_xticks([0,1,2]); ax.set_xticklabels(['$s$','copy of $s$','$u$'],fontsize=8.6); ax.set_ylabel('importance'); grid(ax); sv(fig,'p_25C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

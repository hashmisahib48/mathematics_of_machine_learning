# Problem figures of Chapter 48. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import spearmanr, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import precision_score, recall_score
from sensors import make_sensors
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.5): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]
def cl(m): return 0 if m<=1 else 1 if m==2 else 2*(np.log(m-1)+np.euler_gamma)-2*(m-1)/m
X,Y=make_sensors(); sc=StandardScaler().fit(X); Z=sc.transform(X)

# 48A.1 isolation scores for psi = 16
hs=[2.0,4.0,6.0]; c16=cl(16); sc16=[2**(-h/c16) for h in hs]; A['48A.1']={'c16':round(c16,4),'h':hs,'s':r4(sc16)}
fig,ax=small(2.4,1.5); hh=np.linspace(0,9,200); ax.plot(hh,2**(-hh/c16),color=BLUE,lw=1.4); ax.plot(hs,sc16,'o',color=RED,ms=4.5,zorder=5); ax.axhline(0.5,color=GREEN,lw=0.9,ls='--')
ax.set_xlabel('E $h(\\mathbf{x})$'); ax.set_ylabel('$s$ for $\\psi=16$'); ax.set_ylim(0,1.05); grid(ax); sv(fig,'p_48A1.png')

# 48A.2 LOF by hand
xs=np.array([0,1,2,4,10.])[:,None]; l=LocalOutlierFactor(n_neighbors=2).fit(xs); lv=-l.negative_outlier_factor_
A['48A.2']={'x':xs[:,0].tolist(),'lof':r4(lv)}
fig,ax=small(2.4,1.0); ax.axhline(0,color=GREY,lw=0.8); ax.scatter(xs[:,0],np.zeros(5),s=30,color=BLUE,zorder=3)
for v in xs[:,0]: ax.text(v,0.25,f'{v:.0f}',ha='center',fontsize=9,color=NAVY)
ax.set_xlim(-1,11); ax.set_ylim(-0.4,1); ax.axis('off'); sv(fig,'p_48A2.png')

# 48A.3 Mahalanobis distance
mu=np.array([70,2.5]); S=np.array([[16,1.2],[1.2,0.25]]); q=np.array([78,3.5]); dd=q-mu; d2=float(dd@np.linalg.solve(S,dd)); cut=float(chi2.ppf(0.975,2))
A['48A.3']={'d2':round(d2,4),'euclid2':float(dd@dd),'cut':round(cut,4),'det':float(np.linalg.det(S)),'inv':np.linalg.inv(S).round(4).tolist()}
fig,ax=small(2.4,1.6); w_,v_=np.linalg.eigh(S); th=np.linspace(0,2*np.pi,200)
for r_,ls in ((1,'-'),(np.sqrt(cut),'--')): E_=(v_*np.sqrt(w_))@np.vstack([np.cos(th),np.sin(th)])*r_; ax.plot(mu[0]+E_[0],mu[1]+E_[1],color=GREEN,ls=ls,lw=1.3)
ax.plot(*mu,'o',color=GREEN,ms=3.5); ax.plot(*q,'x',color=RED,ms=6,mew=1.6); ax.set_xlabel('temperature'); ax.set_ylabel('vibration'); grid(ax); sv(fig,'p_48A3.png')

# 48B.1 how many trees
ts=[5,25,100,400]; rho=[]
for t in ts:
    r=[]
    for a in range(8):
        s1=IsolationForest(n_estimators=t,random_state=2*a).fit(Z).score_samples(Z); s2=IsolationForest(n_estimators=t,random_state=2*a+1).fit(Z).score_samples(Z); r.append(spearmanr(s1,s2)[0])
    rho.append((float(np.mean(r)),float(np.std(r))))
A['48B.1']={'trees':ts,'rho':[(round(a,4),round(b,4)) for a,b in rho]}
fig,ax=small(2.4,1.5); ax.errorbar(ts,[a for a,_ in rho],yerr=[b for _,b in rho],fmt='-o',color=BLUE,ms=3.5,lw=1.2,capsize=2); ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(ts); ax.set_xticklabels([str(t) for t in ts]); ax.set_xlabel('trees'); ax.set_ylabel('rank correlation'); grid(ax); sv(fig,'p_48B1.png')

# 48B.2 LOF novelty scores on training rows
lo=LocalOutlierFactor(novelty=True).fit(Z); nov=lo.score_samples(Z); nof=lo.negative_outlier_factor_; fp=(LocalOutlierFactor().fit_predict(Z)==-1).sum(); pr=(lo.predict(Z)==-1).sum()
A['48B.2']={'max_gap':round(float(np.abs(nov-nof).max()),4),'flag_fit_predict':int(fp),'flag_novelty_predict':int(pr),'n_differ':int((np.abs(nov-nof)>1e-9).sum())}
fig,ax=small(2.4,1.5); ax.plot(nof,nov,'.',color=BLUE,ms=3); ax.plot([-5.5,-0.8],[-5.5,-0.8],color=GREEN,lw=1.1); ax.set_xlabel('negative_outlier_factor_',fontsize=8.5); ax.set_ylabel('score_samples'); grid(ax); sv(fig,'p_48B2.png')

# 48B.3 wrong contamination
gs=[0.01,0.02,0.05,0.1,0.2,0.3]; pre=[];rec=[]
for g in gs:
    lab=IsolationForest(contamination=g,random_state=0).fit_predict(Z)==-1; pre.append(precision_score(Y,lab)); rec.append(recall_score(Y,lab))
A['48B.3']={'gamma':gs,'precision':r4(pre),'recall':r4(rec),'n_fault':int(Y.sum()),'n':len(Y)}
fig,ax=small(2.4,1.5); ax.plot(gs,pre,'-o',color=BLUE,ms=3.2,lw=1.2,label='precision'); ax.plot(gs,rec,'-s',color=OR,ms=3.2,lw=1.2,label='recall'); ax.axvline(15/315,color=GREEN,lw=0.9,ls='--'); ax.set_xlabel('contamination'); ax.set_ylim(0,1.08); ax.legend(loc='lower center',handlelength=1.2,labelspacing=0.2,ncol=1,fontsize=8.5); grid(ax); sv(fig,'p_48B3.png')

# 48C.1 a reading between two modes (think; answers not printed)
rng=np.random.RandomState(0); P=np.vstack([rng.normal([-3,0],0.4,(150,2)),rng.normal([3,0],0.4,(150,2))]); mid=np.array([[0.0,0.0]])
ver={}
for nm,d in (('IF',IsolationForest(random_state=0)),('LOF',LocalOutlierFactor(novelty=True)),('OCSVM',OneClassSVM(nu=0.05,gamma=0.1)),('EE',EllipticEnvelope(contamination=0.05,random_state=0))): ver[nm]=int(d.fit(P).predict(mid)[0])
A['_48C.1_check']=ver
fig,ax=small(2.4,1.3); ax.scatter(P[:,0],P[:,1],s=3,color=BLUE,lw=0); ax.plot(0,0,'x',color=RED,ms=7,mew=1.8); ax.set_xlim(-5,5); ax.set_ylim(-2.5,2.5); ax.set_xlabel('feature 1'); ax.set_ylabel('feature 2'); grid(ax); sv(fig,'p_48C1.png')

# 48C.2 base rate
n_=10000; tr=0.002; fl=0.05; A['48C.2']={'n':n_,'true':int(n_*tr),'flagged':int(n_*fl),'max_precision':round(n_*tr/(n_*fl),4)}
fig,ax=small(2.4,1.3); ax.barh([0,1],[n_*tr,n_*fl],color=[RED,BLUE],height=0.55); ax.set_yticks([0,1]); ax.set_yticklabels(['faulty','flagged']); ax.set_xlabel('readings of 10 000'); grid(ax); sv(fig,'p_48C2.png')

for k in [k for k in A if k.startswith('_')]: print(k,A.pop(k))
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

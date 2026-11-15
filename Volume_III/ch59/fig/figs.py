# All figures of Chapter 59. Run: python3 fig/figs.py  (PNGs into ch59/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'xtick.labelsize':8.2,'ytick.labelsize':8.2,'axes.labelsize':8,'legend.fontsize':8.2})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def bx(ax,x,y,w,h,txt,fc=LB,ec=BLUE,fs=8.2,tc=NAVY,bold=False,ls='-'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.05',fc=fc,ec=ec,lw=0.9,ls=ls))
    ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,weight='bold' if bold else 'normal',linespacing=1.15)
def ar(ax,p,q,c=NAVY,lw=0.9): arr(ax,p,q,c=c,lw=lw,ms=7)

# ======== 59.0-1 the procedure in one picture
fig=plt.figure(figsize=(4.6,1.95),dpi=240); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,4.6); ax.set_ylim(0,1.95)
st=[('1  Question','to task','§59.1'),('2  Audit','the data','§59.3, Ch 49–53'),('3  Split','the rows','Ch 55'),('4  Baseline','','Ch 8 (Vol. I)'),
    ('5  Pipeline','','Ch 54'),('6  Compare','and tune','Ch 56, 57'),('7  Threshold','and inspect','Ch 58'),('8  Test once','and report','§59.7')]
bw,bh=1.0,0.74; xs=[0.0,1.2,2.4,3.6]
pos=[]
for k in range(8):
    col=k if k<4 else 7-k; row=0 if k<4 else 1
    x=xs[col]; y=1.15 if row==0 else 0.05
    pos.append((x,y))
    fc=['#E3EEFB','#E3EEFB','#E3EEFB','#E3EEFB','#E6F4EC','#E6F4EC','#E6F4EC','#FDEBD0'][k]; ec=[BLUE]*4+[GREEN]*3+[OR][0:1]
    bx(ax,x,y,bw,bh,'',fc=fc,ec=ec[k] if k<7 else OR)
    ax.text(x+bw/2,y+bh-0.12,st[k][0],ha='center',va='center',fontsize=8.2,color=NAVY,weight='bold')
    ax.text(x+bw/2,y+bh-0.31,st[k][1],ha='center',va='center',fontsize=8.2,color=NAVY)
    ax.text(x+bw/2,y+0.13,st[k][2],ha='center',va='center',fontsize=8.2,color=GREY,style='italic')
for k in range(3): ar(ax,(pos[k][0]+bw+0.01,pos[k][1]+bh/2),(pos[k+1][0]-0.01,pos[k+1][1]+bh/2))
ar(ax,(pos[3][0]+bw/2,pos[3][1]-0.01),(pos[4][0]+bw/2,pos[4][1]+bh+0.01))
for k in range(4,7): ar(ax,(pos[k][0]-0.01,pos[k][1]+bh/2),(pos[k+1][0]+bw+0.01,pos[k+1][1]+bh/2))
# return arrow from 8 up to 2: dashed red, along the left edge
ax.annotate('',xy=(pos[1][0]+0.35,pos[1][1]-0.02),xytext=(pos[7][0]+0.2,pos[7][1]+bh+0.02),arrowprops=dict(arrowstyle='-|>',color=RED,lw=0.9,ls='--',connectionstyle='arc3,rad=0.0',mutation_scale=7))
ax.text(1.62,0.97,'a surprise sends you back',fontsize=8.2,color=RED,ha='left',va='center',style='italic',bbox=dict(fc='white',ec='none',pad=0.8))
sv(fig,'f590.png',440)

# ======== 59.1-1 decision flow chart
fig=plt.figure(figsize=(4.9,3.55),dpi=240); ax=fig.add_axes([0,0,1,1]); ax.axis('off'); ax.set_xlim(0,4.9); ax.set_ylim(0,3.55)
bx(ax,1.45,3.1,2.0,0.38,'Is there a column to predict?',fc=LB,ec=NAVY,bold=True)
ar(ax,(1.9,3.09),(1.25,2.82)); ar(ax,(3.0,3.09),(3.65,2.82)); ax.text(1.4,2.97,'yes',fontsize=8.2,color=GREEN,ha='right',weight='bold'); ax.text(3.5,2.97,'no',fontsize=8.2,color=RED,ha='left',weight='bold')
bx(ax,0.15,2.42,2.2,0.38,'What is the column?',fc='#E6F4EC',ec=GREEN,bold=True)
bx(ax,2.55,2.42,2.2,0.38,'What do you want to learn?',fc='#FDEBD0',ec=OR,bold=True)
L=[('a category','Classification','Ch 16–29',GREEN),('a number','Regression','Ch 9–15, 23, 27, 28',GREEN),('several columns','Multi-output','Ch 33, 35',GREEN),('labels for\nfew rows','Semi-supervised','Ch 34',GREEN)]
R=[('samples that\nbelong together','Clustering','Ch 36–41',OR),('fewer\ncolumns','Reduction','Ch 42–45',OR),('strange\nsamples','Outliers','Ch 48',OR),('the spread\nof the data','Density, covariance','Ch 46, 47',OR)]
for items,x0 in ((L,0.15),(R,2.55)):
    for k,(lab,t1,t2,c_) in enumerate(items):
        y=1.96-k*0.42
        ax.text(x0+0.02,y+0.17,lab,fontsize=8.2,color=NAVY,ha='left',va='center',style='italic',linespacing=0.95)
        bx(ax,x0+0.98,y,1.27,0.34,'',fc='white',ec=c_)
        ax.text(x0+1.04,y+0.23,t1,fontsize=8.2,color=NAVY,weight='bold',va='center'); ax.text(x0+1.04,y+0.09,t2,fontsize=8.2,color=GREY,va='center')
        ar(ax,(x0+0.9,y+0.17),(x0+0.97,y+0.17),c=c_)
bx(ax,0.15,0.02,4.6,0.62,'',fc=LGREY,ec=GRID)
for j,t in enumerate(['Always ask:  are the rows in groups or in time?  (Ch 55)','mixed types, holes, scales?  (Ch 49–54)  ·  which score matters?  (Ch 57)','probabilities or a threshold needed?  (Ch 32, 58)']):
    ax.text(2.45,0.54-j*0.17,t,fontsize=8.2,color=NAVY,ha='center',va='center')
sv(fig,'f591.png',480)

# ======== 59.3-1 the audit of the house table
from house import make_houses
raw=make_houses(); y_=raw['price']
miss=(raw.isna().mean()*100).round(1); 
cols=['area','age','condition','rooms','dist','kind','garden','district']
rho={c:abs(raw[c].corr(y_,method='spearman')) for c in ['area','rooms','age','dist','agent_fee']}
NUM['audit']={'missing':miss.to_dict(),'rho':rho,'skew':float(y_.skew()),'logskew':float(np.log(y_).skew()),'dups':int(raw.duplicated().sum())}
fig,axs=plt.subplots(1,3,figsize=(4.9,1.85),dpi=240,gridspec_kw={'width_ratios':[1.0,1.0,1.0]})
ax=axs[0]; mc=['area','age','condition']; v=[miss[c] for c in mc]
ax.barh(range(3)[::-1],v,color=[BLUE,OR,PUR],height=0.6,edgecolor=NAVY,lw=0.5,hatch=None)
for k,(c,vv) in enumerate(zip(mc,v)): ax.text(vv+0.2,2-k,'%.1f'%vv,va='center',fontsize=8.2,color=NAVY)
ax.set_yticks(range(3)[::-1]); ax.set_yticklabels(mc); ax.set_xlim(0,11.5); ax.set_xlabel('missing (%)'); ax.set_title('(a) holes',fontsize=8,color=NAVY); grid(ax)
ax=axs[1]; names=list(rho); vals=[rho[k] for k in names]
cl=[RED if k=='agent_fee' else BLUE for k in names]
ax.barh(range(5)[::-1],vals,color=cl,height=0.6,edgecolor=NAVY,lw=0.5)
for k,vv in enumerate(vals): ax.text(min(vv+0.03,1.0) if vv<0.9 else vv-0.04,4-k,'%.2f'%vv,va='center',ha='left' if vv<0.9 else 'right',fontsize=8.2,color=NAVY if vv<0.9 else 'white')
ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(['area','rooms','age','dist','agent_fee']); ax.set_xlim(0,1.1); ax.set_xlabel('|Spearman| with price'); ax.set_title('(b) leak screen',fontsize=8,color=NAVY); grid(ax)
ax=axs[2]; ax.hist(y_,bins=40,color=BLUE,edgecolor='white',lw=0.3)
ax.axvline(y_.median(),color=GREEN,lw=1.1,ls='--'); ax.axvline(y_.mean(),color=RED,lw=1.1,ls=':')
ax.set_xlabel('price (thousands)'); ax.set_yticks([]); ax.set_title('(c) target, skew %.2f'%y_.skew(),fontsize=8,color=NAVY)
ax.text(0.97,0.88,'median',color=GREEN,fontsize=8.2,ha='right',transform=ax.transAxes); ax.text(0.97,0.74,'mean',color=RED,fontsize=8.2,ha='right',transform=ax.transAxes)
ax.set_xticks([0,600,1200]); ax.spines['left'].set_visible(False)
fig.tight_layout(w_pad=0.6); sv(fig,'f593.png',430)

# ======== 59.4-1 case A: candidates, threshold, reliability
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_curve
Xb,yb=load_breast_cancer(return_X_y=True)
Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=0.25,random_state=0,stratify=yb)
sc=lambda m: make_pipeline(StandardScaler(),m)
M=[('majority',DummyClassifier(strategy='most_frequent')),('logistic',sc(LogisticRegression(max_iter=1000))),('SVC',sc(SVC())),
   ('forest',RandomForestClassifier(random_state=0)),('hist. boosting',HistGradientBoostingClassifier(random_state=0)),
   ('SVC, tuned',sc(SVC(C=128.0,gamma=2.0**-11)))]
folds={k:cross_val_score(m,Xtr,ytr,cv=5) for k,m in M}
NUM['caseA_cv']={k:[float(v.mean()),float(v.std())] for k,v in folds.items()}
fig,axs=plt.subplots(1,3,figsize=(4.9,1.95),dpi=240,gridspec_kw={'width_ratios':[1.25,1,1]})
ax=axs[0]; ks=list(folds)
for k,name in enumerate(ks):
    yy=len(ks)-1-k; v=folds[name]
    if k==0:
        ax.scatter([0.9265],[yy],s=22,marker='<',color=GREY,zorder=3); ax.text(0.937,yy,'%.3f'%v.mean(),fontsize=8.2,va='center',color=GREY); continue
    ax.scatter(v,[yy]*5,s=9,color=BLUE,zorder=2,alpha=0.8)
    ax.scatter([v.mean()],[yy],s=34,marker='D',color=GREEN if k==5 else NAVY,zorder=3)
ax.set_yticks(range(len(ks))[::-1]); ax.set_yticklabels(ks); ax.set_xlim(0.92,1.008); ax.set_xticks([0.94,0.97,1.0]); ax.set_xticklabels(['0.94','0.97','1.00']); ax.set_xlabel('5-fold accuracy'); grid(ax); ax.set_title('(a) candidates',fontsize=8,color=NAVY)
svc=sc(SVC(C=128.0,gamma=2.0**-11)); cal=CalibratedClassifierCV(svc,method='sigmoid',cv=5)
oof=cross_val_predict(cal,Xtr,ytr,cv=5,method='predict_proba')[:,0]; mal=ytr==0
fpr,tpr,thr=roc_curve(mal,oof); t=float(thr[np.argmax(tpr>=0.99)])
NUM['caseA_t']=t
ax=axs[1]; ok=np.isfinite(thr); 
ax.plot(thr[ok],tpr[ok],color=GREEN,lw=1.4,label='recall'); ax.plot(thr[ok],fpr[ok],color=RED,lw=1.4,ls='--',label='false alarms')
ax.axvline(t,color=PUR,lw=1.0); ax.axvline(0.5,color=GREY,lw=1.0,ls=':'); ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.set_xlabel('threshold'); ax.set_yticks([0,0.5,1]); grid(ax)
ax.text(t+0.03,0.70,'%.3f'%t,color=PUR,fontsize=8.2,rotation=90,va='center'); ax.text(0.72,0.90,'recall',color=GREEN,fontsize=8.2); ax.text(0.22,0.30,'false\nalarms',color=RED,fontsize=8.2,linespacing=1.0); ax.set_title('(b) out of fold',fontsize=8,color=NAVY)
ax=axs[2]; fr,mp=calibration_curve(mal,oof,n_bins=5,strategy='quantile'); NUM['caseA_rel']=[mp.tolist(),fr.tolist()]
ax.plot([0,1],[0,1],color=GREY,lw=0.9,ls=':'); ax.plot(mp,fr,color=GREEN,marker='o',ms=4,lw=1.3); ax.set_xlabel('forecast P(malignant)'); ax.set_ylabel('observed share'); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); grid(ax); ax.set_title('(c) reliability',fontsize=8,color=NAVY)
fig.tight_layout(w_pad=0.5); sv(fig,'f594.png',430)

# ======== 59.5-1 case B: candidates, predictions, importance
import re
from case_b1 import Xtr as hXtr, ytr as hytr, test as htest, hgb, log, models
htab=[]
for line in open('verify/case_b1.out').read().splitlines():
    m=re.match(r'(.{20})R2\s+(-?[\d.]+)\s+MAE\s+([\d.]+)',line); htab.append((m.group(1).strip(),float(m.group(2)),float(m.group(3))))
imp=[]
for line in open('verify/case_b2.out').read().splitlines()[4:]:
    m=re.match(r'\s+(\w+)\s+([\d.]+) \+- ([\d.]+)',line)
    if m: imp.append((m.group(1),float(m.group(2)),float(m.group(3))))
NUM['caseB']={'cv':htab,'imp':imp}
final=log(hgb()).fit(hXtr,hytr); Xh=htest.drop(columns='price'); yh=htest['price'].values; ph=final.predict(Xh)
fig,axs=plt.subplots(1,3,figsize=(4.9,1.95),dpi=240,gridspec_kw={'width_ratios':[1.25,1,1.0]})
ax=axs[0]; nm=['mean','ridge','ridge, log','boosting','boosting, log']
cols=[GREY,BLUE,BLUE,GREEN,GREEN]; hat=['','','//','','//']
ax.barh(range(5)[::-1],[h[2] for h in htab],color=cols,edgecolor=NAVY,lw=0.5,height=0.68)
for b_,h_ in zip(ax.patches,hat): b_.set_hatch(h_)
for k,h in enumerate(htab): ax.text(h[2]+2,4-k,'%.0f'%h[2],va='center',fontsize=8.2,color=NAVY)
ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(nm,fontsize=8.2); ax.set_xlim(0,128); ax.set_xticks([0,50,100]); ax.set_xlabel('5-fold MAE'); ax.set_title('(a) candidates',fontsize=8,color=NAVY); grid(ax)
ax=axs[1]; ax.scatter(yh,ph,s=3,color=BLUE,alpha=0.5,lw=0); lim=[40,1000]; ax.plot(lim,lim,color=GREY,lw=0.9,ls=':')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([100,1000]); ax.set_xticklabels(['100','1000']); ax.set_yticks([100,1000]); ax.set_yticklabels(['100','1000']); ax.minorticks_off()
ax.set_xlabel('actual price'); ax.set_ylabel('predicted'); ax.set_title('(b) test rows',fontsize=8,color=NAVY); grid(ax)
ax=axs[2]; im=imp[::-1]
ax.barh(range(len(im)),[v[1] for v in im],xerr=[v[2] for v in im],color=GREEN,height=0.65,edgecolor=NAVY,lw=0.5,error_kw=dict(lw=0.8,ecolor=NAVY,capsize=1.5))
ax.set_yticks(range(len(im))); ax.set_yticklabels([v[0] for v in im],fontsize=8.2); ax.set_xlabel('rise of MAE'); ax.set_title('(c) shuffling a column',fontsize=8,color=NAVY); grid(ax)
fig.tight_layout(w_pad=0.5); sv(fig,'f595.png',430)

# ======== 59.6-1 case C: wine
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score as sil, adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture
Xw,yw=load_wine(return_X_y=True); Xs=StandardScaler().fit_transform(Xw); Z=PCA(2,random_state=0).fit_transform(Xs)
Ks=list(range(2,7)); sl=[]; bc=[]
for k in Ks:
    l=KMeans(k,n_init=10,random_state=0).fit_predict(Z); sl.append(sil(Z,l)); bc.append(GaussianMixture(k,n_init=5,random_state=0).fit(Z).bic(Z))
km=KMeans(3,n_init=10,random_state=0).fit(Z); lab=km.labels_
# colour = cluster, shape = cultivar
cc=[BLUE,OR,GREEN]; mk=['o','s','^']
NUM['caseC']={'sil':sl,'bic':bc,'ari':float(ari(yw,lab))}
fig,axs=plt.subplots(1,3,figsize=(4.9,1.95),dpi=240,gridspec_kw={'width_ratios':[1.1,1,1.1]})
ax=axs[0]
for c_ in range(3):
    for g in range(3):
        m_=(lab==c_)&(yw==g)
        if m_.any(): ax.scatter(Z[m_,0],Z[m_,1],s=9,color=cc[c_],marker=mk[g],lw=0.3,edgecolor='white',alpha=0.9)
ax.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],s=45,marker='X',color=NAVY,zorder=4,edgecolor='white',lw=0.5)
ax.set_xlabel('PC 1 (36 %)'); ax.set_ylabel('PC 2 (19 %)'); ax.set_xticks([-2,0,2,4]); ax.set_yticks([-2,0,2]); ax.set_title('(a) colour: cluster',fontsize=8,color=NAVY); grid(ax)
ax=axs[1]; ax.plot(Ks,sl,color=GREEN,marker='o',ms=3.5,lw=1.3); ax.set_xlabel('K'); ax.set_ylabel('silhouette',color=GREEN); ax.set_xticks(Ks); ax.set_ylim(0.40,0.60); ax.set_yticks([0.4,0.5,0.6]); ax.tick_params(axis='y',colors=GREEN)
ax2=ax.twinx(); ax2.plot(Ks,bc,color=PUR,marker='s',ms=3.5,lw=1.3,ls='--'); ax2.set_ylabel('BIC',color=PUR); ax2.tick_params(axis='y',colors=PUR,labelsize=8.2); ax2.spines['right'].set_visible(True); ax2.set_ylim(1300,1380); ax2.set_yticks([1300,1340,1380])
ax.axvline(3,color=GREY,lw=0.8,ls=':'); ax.set_title('(b) K by two scores',fontsize=8,color=NAVY); grid(ax)
T=TSNE(2,perplexity=30,init='pca',random_state=0).fit_transform(Xs)
ax=axs[2]
for c_ in range(3):
    for g in range(3):
        m_=(lab==c_)&(yw==g)
        if m_.any(): ax.scatter(T[m_,0],T[m_,1],s=9,color=cc[c_],marker=mk[g],lw=0.3,edgecolor='white',alpha=0.9)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2'); ax.set_title('(c) t-SNE, same marks',fontsize=8,color=NAVY)
fig.tight_layout(w_pad=0.5); sv(fig,'f596.png',430)

# ======== 59.7-1 reporting: repeated CV and the width of a test score
from sklearn.model_selection import RepeatedStratifiedKFold
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=0)
cand=[('logistic',sc(LogisticRegression(max_iter=1000))),('SVC',sc(SVC())),('forest',RandomForestClassifier(random_state=0))]
rng=np.random.RandomState(1)
H=json.load(open('verify/hand.json'))
fig,axs=plt.subplots(1,2,figsize=(4.9,1.75),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; SS={}
for k,(nm_,m_) in enumerate(cand):
    sc_=cross_val_score(m_,Xtr,ytr,cv=cv); SS[nm_]=sc_.tolist(); yy=2-k
    ax.scatter(sc_,yy+rng.uniform(-0.22,0.22,50),s=4,color=BLUE,alpha=0.55,lw=0)
    rm=sc_.reshape(10,5).mean(axis=1); ax.scatter(rm,[yy]*10,s=16,marker='D',color=OR,edgecolor=NAVY,lw=0.4,zorder=3)
ax.set_yticks([2,1,0]); ax.set_yticklabels([c[0] for c in cand]); ax.set_xlabel('accuracy, 5 folds × 10 repeats'); ax.set_xlim(0.90,1.005); ax.set_xticks([0.92,0.96,1.0]); ax.set_title('(a) 50 scores, 10 repeat means',fontsize=8,color=NAVY); grid(ax)
NUM['repeats']={k:[float(np.mean(v)),float(np.std(v))] for k,v in SS.items()}
ax=axs[1]; A=H['acc']; ivs=[('normal',A['lo'],A['hi']),('Wilson',A['wilson'][0],A['wilson'][1]),('bootstrap',A['boot'][0],A['boot'][1])]
for k,(nm_,lo,hi) in enumerate(ivs):
    yy=2-k; ax.plot([lo,hi],[yy,yy],color=GREEN,lw=2.4,solid_capstyle='butt'); ax.scatter([A['acc']],[yy],s=14,color=NAVY,zorder=3)
rf=RandomForestClassifier(random_state=0).fit(Xtr,ytr).score(Xte,yte); hg=HistGradientBoostingClassifier(random_state=0).fit(Xtr,ytr).score(Xte,yte); NUM['other_test']={'forest':rf,'hgb':hg}
for v_,t_,ha_ in ((rf,'forest','right'),(hg,'boosting','right')):
    ax.axvline(v_,color=GREY,lw=0.8,ls=':'); 
ax.text(rf+0.002,-0.62,'forest',fontsize=8.2,color=GREY,ha='left'); ax.text(hg-0.002,-0.62,'boosting',fontsize=8.2,color=GREY,ha='right')
ax.set_yticks([2,1,0]); ax.set_yticklabels([i_[0] for i_ in ivs]); ax.set_ylim(-0.9,2.5); ax.set_xlim(0.90,1.0); ax.set_xticks([0.92,0.96,1.0]); ax.set_xlabel('test accuracy, 143 patients'); ax.set_title('(b) 95 % intervals',fontsize=8,color=NAVY); grid(ax)
fig.tight_layout(w_pad=0.6); sv(fig,'f597.png',430)
json.dump({'NUM':NUM,'CHK':CHK},open('verify/fig_numbers.json','w'),indent=1,default=float)

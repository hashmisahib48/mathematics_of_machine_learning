# All figures of Chapter 58. Run: python3 fig/figs.py (numbers into verify/fig_numbers.json)
import sys,os,json,time,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.special import logit, expit
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingClassifier as GBC, RandomForestClassifier as RF
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bcdata import load
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
Xa,Xb,ya,yb,names=load(); names=[str(a) for a in names]; ym,yt=1-ya,1-yb
lr=make_pipeline(StandardScaler(),LogisticRegression())
def cost(y,pred,cfn,cfp,pos_label):
    miss=(y==pos_label)&(pred!=pos_label); alarm=(y!=pos_label)&(pred==pos_label)
    return (cfn*miss.sum()+cfp*alarm.sum())/len(y)
def tuned(r,cv=5):
    sc=make_scorer(cost,greater_is_better=False,cfn=r,cfp=1,pos_label=0)
    return TunedThresholdClassifierCV(lr,scoring=sc,cv=cv,store_cv_results=True).fit(Xa,ya)
lr.fit(Xa,ya); st_test=lr.decision_function(Xb); pm_logit=-st_test   # log-odds of malignant
t0=time.time()

# ======== 58.2-1 the threshold slides along the log-odds
fig,ax=plt.subplots(figsize=(4.0,1.75),dpi=240)
bins=np.linspace(-12,12,25)
ax.hist([pm_logit[yb==0],pm_logit[yb==1]],bins=bins,stacked=True,color=[OR,BLUE],edgecolor='white',lw=0.4,label=['malignant (53)','benign (90)'])
ts=[(0.5,GREY,'-'),(0.2,PUR,'--'),(0.05,GREEN,':')]
for t,c_,ls in ts: ax.axvline(logit(t),color=c_,ls=ls,lw=1.6)
ax.set_xlim(-12,12); ax.set_ylim(0,47)
for t,c_,ls in ts: ax.text(logit(t)+(0.45 if t==0.5 else -0.45),30,'$t$ = %g'%t,color=c_,fontsize=8.5,ha='center',va='center',rotation=90)
pt=[0.0001,0.01,0.2,0.5,0.8,0.99,0.9999]; ax.set_xticks(logit(pt)); ax.set_xticklabels(['0.0001','0.01','0.2','0.5','0.8','0.99','0.9999'])
ax.set_xlabel('predicted probability of malignancy (scale: log-odds)'); ax.set_ylabel('test patients'); ax.legend(loc='center left',bbox_to_anchor=(0.0,0.62),fontsize=8.5,handlelength=1,borderaxespad=0.1); grid(ax)
fig.tight_layout(pad=0.3); sv(fig,'f582.png',380)
cnt={}
for t,_,_ in ts:
    yes=expit(pm_logit)>=t; cnt[t]={'caught':int((yes&(yb==0)).sum()),'missed':int((~yes&(yb==0)).sum()),'false_alarms':int((yes&(yb==1)).sum())}
NUM['slide']=cnt

# ======== 58.3-1 expected cost of the two actions
fig,ax=plt.subplots(figsize=(2.9,2.15),dpi=240)
p=np.linspace(0,0.5,200)
ax.plot(p,1-p,color=BLUE,lw=1.6,label='say yes: $(1-p)\\,c_{FP}$')
for cfn,ls in ((5,'-'),(20,'--')):
    ax.plot(p,cfn*p,color=RED,lw=1.5,ls=ls,label='say no: $p\\,c_{FN}$ ($c_{FN}$ = %d)'%cfn); t=1/(1+cfn); ax.plot([t],[1-t],'o',color=GREEN,ms=5,zorder=5)
    ax.plot([t,t],[0,1-t],color=GREEN,lw=0.8,ls=':')
ax.legend(loc='lower right',fontsize=8.2,handlelength=1.6,borderaxespad=0.1,labelspacing=0.25)
ax.set_xticks([0,1/21,1/6,0.3,0.4,0.5]); ax.set_xticklabels(['0','1/21','1/6','0.3','0.4','0.5'])
for tl,c_ in zip(ax.get_xticklabels(),[GREY,GREEN,GREEN,GREY,GREY,GREY]): tl.set_color(c_)
ax.set_xlim(0,0.5); ax.set_ylim(0,1.4); ax.set_xlabel('probability of malignancy $p$'); ax.set_ylabel('expected cost'); grid(ax)
fig.tight_layout(pad=0.3); sv(fig,'f583.png',270)

# ======== 58.7-1 the two loops
fig,ax=plt.subplots(figsize=(4.6,2.25),dpi=240); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(0,5)
def bx(x,y,w,h,t,fc=LB,ec=BLUE): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=7.6,r=0.06)
def ar(x0,y0,x1,y1): arr(ax,(x0,y0),(x1,y1),c=GREY,lw=0.9,ms=7)
ax.text(0.0,4.78,'Tuned threshold',fontsize=9,color=NAVY,weight='bold',va='center'); ax.text(0.0,2.55,'Permutation importance',fontsize=9,color=NAVY,weight='bold',va='center')
w=2.3; xs=[0.0,2.55,5.1,7.65]
lab=['split the rows\ninto $K$ folds;\nfit on $K-1$','score the held-out\nfold at 100\nthresholds','interpolate the $K$\ncurves to one grid;\naverage','$\\tau$ = argmax;\nrefit on all\nrows']
for k,(x,t) in enumerate(zip(xs,lab)):
    bx(x,3.2,w,1.3,t,fc=LB if k<3 else '#E3F4EA',ec=BLUE if k<3 else GREEN)
    if k<3: ar(x+w,3.85,xs[k+1],3.85)
lab2=['baseline score\n$s_0$ on $(X,y)$','shuffle column $j$\n(repeat $R$ times)','score again:\n$s_{jr}$','$I_{j}=s_0-\\overline{s_{jr}}$']
xs2=[0.0,2.55,5.1,7.65]; w2=2.3
for k,(x,t) in enumerate(zip(xs2,lab2)):
    bx(x,0.95,w2,1.05,t,fc=LB if k<3 else '#E3F4EA',ec=BLUE if k<3 else GREEN)
    if k<3: ar(x+w2,1.47,xs2[k+1],1.47)
ax.text(0.0,0.4,'Partial dependence: for each grid value $v$, set column $j$ to $v$ in every row, predict, average.',fontsize=7.8,color=NAVY,va='center')
sv(fig,'f586.png',440)

# ======== 58.9-1 the knob: the cost ratio c_FN / c_FP
R=[1,5,10,20]; res={}
for r in R:
    t=tuned(r); res[r]=(t.cv_results_['thresholds'],-t.cv_results_['scores'],t.best_threshold_)
seeds=range(20); spread={r:[] for r in R}
for r in R:
    for sd in seeds:
        spread[r].append(tuned(r,cv=StratifiedKFold(5,shuffle=True,random_state=sd)).best_threshold_)
NUM['knob']={'tuned_seed0':{r:round(float(res[r][2]),4) for r in R},'tstar':{r:round(1/(1+r),4) for r in R},
  'spread_min':{r:round(float(min(spread[r])),4) for r in R},'spread_max':{r:round(float(max(spread[r])),4) for r in R},'spread_median':{r:round(float(np.median(spread[r])),4) for r in R},
  'spread_all':{r:[round(float(v),4) for v in spread[r]] for r in R}}
fig,axs=plt.subplots(1,2,figsize=(4.4,1.8),dpi=240,gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]; sty={1:('-',GREY),5:('--',PUR),10:('-.',BLUE),20:(':',GREEN)}
for r in R:
    th,cc,b=res[r]; ls,col=sty[r]; ax.plot(th,cc,ls=ls,color=col,lw=1.5,label='$c_{FN}/c_{FP}$ = %d'%r); ax.plot([b],[cc[np.argmin(np.abs(th-b))]],'o',color=col,ms=4.5,zorder=5)
ax.set_xscale('log'); ax.set_xlim(0.004,1); ax.set_ylim(0,0.45); ax.set_xlabel('threshold $\\tau$ on $P$(malignant)'); ax.set_ylabel('mean CV cost per patient'); ax.legend(loc='upper left',ncol=2,fontsize=7.4,handlelength=1.5,columnspacing=0.8,borderaxespad=0.1,labelspacing=0.2); grid(ax)
ax=axs[1]; xs=np.arange(len(R))
for k,r in enumerate(R):
    v=np.array(spread[r]); ax.plot(np.full(len(v),k)+np.linspace(-0.18,0.18,len(v)),v,'o',color=PUR,ms=2.8,mfc='none',mew=0.7)
    ax.plot([k-0.3,k+0.3],[1/(1+r)]*2,color=GREEN,lw=1.8)
ax.set_xticks(xs); ax.set_xticklabels([str(r) for r in R]); ax.set_xlabel('$c_{FN}/c_{FP}$'); ax.set_ylabel('best threshold'); ax.set_ylim(0,0.6)
ax.legend(handles=[Line2D([],[],color=GREEN,lw=1.8,label='$t^{*}$, Eq. 58.3-2'),Line2D([],[],marker='o',ls='',mfc='none',color=PUR,ms=3.5,label='20 fold seeds')],loc='upper right',fontsize=7.8,handlelength=1.2,borderaxespad=0.1,labelspacing=0.2); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f5891.png',420)

# ======== 58.9-2 three importances on a noisy set; a correlated block
def make(n,seed):
    r=np.random.RandomState(seed); sg=r.standard_normal(n); N=r.standard_normal((n,9)); y=(sg+1.0*r.standard_normal(n)>0).astype(int); return np.c_[sg,N],y
Xs,ys=make(200,0); Xt,yt_=make(1000,1)
rf=RF(n_estimators=200,random_state=0).fit(Xs,ys)
imp=rf.feature_importances_; ptr=permutation_importance(rf,Xs,ys,n_repeats=20,random_state=0).importances_mean; pte=permutation_importance(rf,Xt,yt_,n_repeats=20,random_state=0).importances_mean
NUM['synthetic']={'train_acc':round(rf.score(Xs,ys),4),'test_acc':round(rf.score(Xt,yt_),4),'impurity':[round(float(v),4) for v in imp],'perm_train':[round(float(v),4) for v in ptr],'perm_test':[round(float(v),4) for v in pte],
  'noise_share_impurity':round(float(imp[1:].sum()),4),'noise_sum_train':round(float(ptr[1:].sum()),4),'noise_sum_test':round(float(pte[1:].sum()),4)}
R_=np.abs(np.corrcoef(Xa,rowvar=False)); cl=fcluster(linkage(squareform(1-R_,checks=False),'average'),0.1,'distance')
block=[j for j in range(30) if cl[j]==cl[0]]; NUM['block']=[names[j] for j in block]
fig,axs=plt.subplots(1,3,figsize=(4.4,1.45),dpi=240)
for k,(ax,v,tt) in enumerate(zip(axs,(imp,ptr,pte),('impurity','permutation, training','permutation, test'))):
    ax.bar(np.arange(10),v,color=[BLUE]+[LGREY]*9,edgecolor=[BLUE]+[GREY]*9,lw=0.5,width=0.8); ax.axhline(0,color=GREY,lw=0.6)
    ax.set_xticks([0,5]); ax.set_xticklabels(['signal','noise ×9'],fontsize=8); ax.tick_params(axis='x',length=0)
    ax.set_title(tt,fontsize=8.5,color=NAVY,pad=3); ax.set_ylim(-0.03,0.42)
    if k==0: ax.set_ylabel('importance')
    grid(ax)
fig.tight_layout(w_pad=0.6); sv(fig,'f5892.png',420)
def blockdrop(mdl,Xe,ye,reps=30):
    rng=np.random.RandomState(0); base=mdl.score(Xe,ye); out=[]
    for _ in range(reps):
        Xp=Xe.copy(); Xp[:,block]=Xe[rng.permutation(len(ye))][:,block]; out.append(base-mdl.score(Xp,ye))
    return float(np.mean(out))
gbm=GBC(random_state=0).fit(Xa,ym); rfb=RF(n_estimators=200,random_state=0).fit(Xa,ym)
for mdl,tt in ((rfb,'forest'),(gbm,'boosting')):
    pi=permutation_importance(mdl,Xb,yt,n_repeats=30,random_state=0).importances_mean; one=pi[block]; gd=blockdrop(mdl,Xb,yt)
    NUM['block_'+tt]={'singles':[round(float(v),4) for v in one],'sum':round(float(one.sum()),4),'max':round(float(one.max()),4),'group':round(gd,4),'test_acc':round(mdl.score(Xb,yt),4)}

# ======== 58.10-1 case study: partial dependence of the boosting model
gbm=GBC(random_state=0).fit(Xa,ym)
feats=[21,27,22,20]; fig,axs=plt.subplots(1,4,figsize=(4.7,1.6),dpi=240,sharey=True); rg=np.random.RandomState(1); sel=rg.choice(len(yt),40,replace=False); NUM['case_pd']={}
for ax,j in zip(axs,feats):
    pdj=partial_dependence(gbm,Xb,[j],kind='both',grid_resolution=50,method='brute',response_method='predict_proba'); gj=pdj['grid_values'][0]
    for i_ in sel: ax.plot(gj,pdj['individual'][0][i_],color=OR if yt[i_]==1 else BLUE,lw=0.4,alpha=0.5)
    ax.plot(gj,pdj['average'][0],color=GREEN,lw=2.2); ax.set_title(names[j].replace('worst ','worst\n'),fontsize=8.5,color=NAVY,pad=2); ax.set_xticks(np.round(np.linspace(gj[0],gj[-1],3),2 if gj[-1]<1 else 0)); ax.tick_params(labelsize=8,pad=1.5); grid(ax)
    NUM['case_pd'][names[j]]={'pd_first':round(float(pdj['average'][0][0]),4),'pd_last':round(float(pdj['average'][0][-1]),4),'grid':[round(float(gj[0]),3),round(float(gj[-1]),3)]}
axs[0].set_ylabel('$P$(malignant)',fontsize=8.5); axs[0].set_ylim(-0.03,1.03)
fig.tight_layout(w_pad=0.3); sv(fig,'f5810.png',450)

# ======== 58.0-1 opener
kw=dict(n_repeats=30,random_state=0); ll=permutation_importance(gbm,Xb,yt,scoring='neg_log_loss',**kw)
o=np.argsort(-ll.importances_mean)[:6]
fig,axs=plt.subplots(1,2,figsize=(4.3,1.62),dpi=240,gridspec_kw={'width_ratios':[1,1.05]})
ax=axs[0]; th,cc,b=res[10]; ax.plot(th,cc,color=BLUE,lw=1.6); ax.set_xscale('log'); ax.set_xlim(0.004,1); ax.set_ylim(0,0.45)
for tv,lab,col,mk in ((0.5,'default 0.5',GREY,'s'),(1/11,'$t^{*}$ = 1/11',GREEN,'^'),(b,'tuned %.2f'%b,PUR,'o')):
    yv=cc[np.argmin(np.abs(th-tv))]; ax.plot([tv],[yv],mk,color=col,ms=5.5,zorder=5)
ax.text(0.5,0.30,'0.5',color=GREY,fontsize=8.2,ha='right'); ax.text(1/11,0.085,'1/11',color=GREEN,fontsize=8.2,ha='center',va='top'); ax.text(b*1.15,0.03,'%.2f'%b,color=PUR,fontsize=8.2,ha='left',va='bottom')
ax.set_xlabel('threshold $\\tau$ on $P$(malignant)'); ax.set_ylabel('mean CV cost per patient'); ax.set_title('cost ratio $c_{FN}/c_{FP}=10$',fontsize=8.5,color=NAVY,pad=3); grid(ax)
ax=axs[1]; lab=[names[j] for j in o]; yy=np.arange(len(o))[::-1]
ax.barh(yy,ll.importances_mean[o],xerr=ll.importances_std[o],color=BLUE,height=0.65,error_kw=dict(lw=0.8,ecolor=GREY,capsize=1.5)); ax.set_yticks(yy); ax.set_yticklabels(lab,fontsize=8.2); ax.set_xlabel('rise in log loss when shuffled'); ax.set_title('what the model uses',fontsize=8.5,color=NAVY,pad=3); grid(ax)
fig.tight_layout(w_pad=0.5); sv(fig,'f580.png',410)
NUM['opener']={'top':[names[j] for j in o],'imp':[round(float(ll.importances_mean[j]),4) for j in o],'sd':[round(float(ll.importances_std[j]),4) for j in o]}
print('fig basics',round(time.time()-t0,1)); sys.stdout.flush()
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)

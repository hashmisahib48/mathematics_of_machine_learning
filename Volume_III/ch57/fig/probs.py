# Problem figures of Chapter 57. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import norm
from sklearn import metrics as M
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from screendata import make_screen
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
A={}; CHK={}
def small(w=1.85,h=0.8): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*115,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def cm_fig(ax,TP,FN,FP,TN):
    for (r_,c_),v,fc,ec in (((1,0),TP,LB,BLUE),((1,1),FN,PINK,RED),((0,0),FP,PINK,RED),((0,1),TN,LB,BLUE)):
        ax.add_patch(Rectangle((c_,r_),1,1,fc=fc,ec=ec,lw=0.9)); ax.text(c_+0.5,r_+0.5,str(v),ha='center',va='center',fontsize=8.5,color=NAVY,weight='bold')
    ax.set_xlim(-0.55,2); ax.set_ylim(0,2); ax.axis('off')
    ax.text(0.5,2.12,'flagged',ha='center',fontsize=7.8,color=NAVY); ax.text(1.5,2.12,'not',ha='center',fontsize=7.8,color=NAVY)
    ax.text(-0.05,1.5,'ill',ha='right',va='center',fontsize=7.8,color=NAVY); ax.text(-0.05,0.5,'well',ha='right',va='center',fontsize=7.8,color=NAVY)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 57A.1 confusion counts to scores
TP,FN,FP,TN=30,10,20,940
y=np.r_[np.ones(TP+FN),np.zeros(FP+TN)].astype(int); p=np.r_[np.ones(TP),np.zeros(FN),np.ones(FP),np.zeros(TN)].astype(int)
A['57A.1']={'precision':round(M.precision_score(y,p),4),'recall':round(M.recall_score(y,p),4),'F1':round(M.f1_score(y,p),4),'F2':round(M.fbeta_score(y,p,beta=2),4),'MCC':round(M.matthews_corrcoef(y,p),4),'accuracy':round(M.accuracy_score(y,p),4)}
fig,ax=small(1.85,0.8); cm_fig(ax,TP,FN,FP,TN); sv(fig,'p_57A1.png')

# 57A.2 AUC and AP of six scores
y6=np.array([1,0,1,0,1,0]); s6=np.array([.8,.7,.6,.4,.3,.2]); f,t,_=M.roc_curve(y6,s6,drop_intermediate=False)
A['57A.2']={'AUC':round(M.roc_auc_score(y6,s6),4),'AP':round(M.average_precision_score(y6,s6),4),'pairs_won_of_9':6}
fig,ax=small(1.85,0.8); ax.fill_between(f,t,color=GREEN,alpha=0.18,lw=0); ax.plot(f,t,'-o',color=GREEN,lw=1.2,ms=2.8); ax.plot([0,1],[0,1],color=GREY,lw=0.6,ls=':')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); sv(fig,'p_57A2.png')

# 57A.3 positive predictive value
se,sp=0.9,0.95; ppv=lambda pi: se*pi/(se*pi+(1-sp)*(1-pi))
A['57A.3']={'PPV_prevalence_1%':round(ppv(0.01),4),'PPV_prevalence_10%':round(ppv(0.10),4),'PPV_prevalence_50%':round(ppv(0.5),4),'LR+':round(se/(1-sp),2)}
fig,ax=small(1.85,0.8); pis=np.logspace(-3,np.log10(0.5),100); ax.plot(pis,ppv(pis),color=PUR,lw=1.3); ax.set_xscale('log'); ax.set_xlabel('prevalence'); ax.set_ylabel('precision'); ax.set_ylim(0,1.02); ax.set_yticks([0,0.5,1]); grid(ax)
for pi in (0.01,0.1): ax.plot(pi,ppv(pi),'o',color=RED,ms=3.5)
sv(fig,'p_57A3.png')

# 57B.1 AUC of every single feature
X,yb=load_breast_cancer(return_X_y=True); names=load_breast_cancer().feature_names; ym=1-yb
au=np.array([M.roc_auc_score(ym,X[:,j]) for j in range(30)]); best=int(np.argmax(np.maximum(au,1-au)))
A['57B.1']={'best_feature':str(names[best]),'AUC_best':round(float(max(au[best],1-au[best])),4),'n_features_with_AUC_below_0.5':int((au<0.5).sum()),'n_features_with_AUC_above_0.9':int((np.maximum(au,1-au)>0.9).sum()),'AUC_of_worst_feature_as_given':round(float(au.min()),4),'smallest_distance_from_0.5':round(float(np.abs(au-0.5).min()),4)}
fig,ax=small(1.85,0.8); ax.hist(np.maximum(au,1-au),bins=np.linspace(0.5,1,11),color=BLUE,edgecolor='white'); ax.set_xlabel('AUC of one feature'); ax.set_ylabel('features'); ax.set_xlim(0.5,1); sv(fig,'p_57B1.png')

# 57B.2 F2 scorer with pos_label
Xtr,Xte,ytr,yte=train_test_split(X,yb,test_size=0.25,random_state=0,stratify=yb)
sc=M.make_scorer(M.fbeta_score,beta=2,pos_label=0); cv=StratifiedKFold(5,shuffle=True,random_state=0)
lr=make_pipeline(StandardScaler(),LogisticRegression()); rf=RandomForestClassifier(random_state=0)
r_lr=cross_val_score(lr,Xtr,ytr,cv=cv,scoring=sc).mean(); r_rf=cross_val_score(rf,Xtr,ytr,cv=cv,scoring=sc).mean()
A['57B.2']={'F2_malignant_logistic':round(r_lr,4),'F2_malignant_forest':round(r_rf,4)}
fig,ax=small(1.85,0.8); ax.bar([0,1],[r_lr,r_rf],0.5,color=[GREEN,RED]); ax.set_xticks([0,1]); ax.set_xticklabels(['logistic','forest']); ax.set_ylim(0.85,1.0); ax.set_ylabel('F2'); ax.set_yticks([0.9,0.95,1.0])
for k,v in enumerate([r_lr,r_rf]): ax.text(k,v+0.004,'%.3f'%v,ha='center',fontsize=7.8,color=NAVY)
sv(fig,'p_57B2.png')

# 57B.3 clipping a forest's probabilities
rf.fit(Xtr,ytr); P=rf.predict_proba(Xte)[:,1]; eps=[0,0.005,0.01,0.02,0.05]
ll=[M.log_loss(yte,np.clip(P,e,1-e)) for e in eps]
A['57B.3']={'epsilon':eps,'log_loss':r4(ll),'n_exact_0_or_1':int(((P==0)|(P==1)).sum()),'n_wrong_at_exact_0_or_1':int((((P==0)&(yte==1))|((P==1)&(yte==0))).sum())}
fig,ax=small(1.85,0.8); ax.plot(eps,ll,'-o',color=RED,lw=1.2,ms=3); ax.set_xlabel('clip level ε'); ax.set_ylabel('log loss'); ax.set_xticks([0,0.025,0.05]); grid(ax); sv(fig,'p_57B3.png')

# 57C.1 AUC 0.99 and recall 0.5
fig,ax=small(1.85,0.8); tt=np.linspace(-6,8,500); fp_=norm.sf(tt); tp_=norm.sf(tt-3.3)
ax.plot(fp_,tp_,color=GREEN,lw=1.3); ax.plot(norm.sf(3.3),0.5,'o',color=RED,ms=4); ax.text(0.04,0.37,'recall 0.5',fontsize=7.8,color=RED); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); grid(ax); sv(fig,'p_57C1.png')

# 57C.2 same ROC, two prevalences
fig,ax=small(1.85,0.8)
for pi,col,ls in ((0.5,BLUE,'-'),(0.05,RED,'--')):
    s_,y_=make_screen(100000,pi,2.0,3); p_,r_,_=M.precision_recall_curve(y_,s_); ax.plot(r_,p_,color=col,ls=ls,lw=1.2,label='π = %g'%pi)
ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_ylim(0,1.02); ax.set_yticks([0,0.5,1]); ax.legend(loc='lower left',fontsize=7.8,handlelength=1.5,borderaxespad=0.1); grid(ax); sv(fig,'p_57C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/problem_font_check.json','w'),indent=1)
print(json.dumps(A,indent=0)); print(CHK)

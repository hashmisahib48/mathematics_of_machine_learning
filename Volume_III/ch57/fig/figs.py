# All figures of Chapter 57. Run: python3 fig/figs.py  (PNGs into ch57/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import norm
from sklearn import metrics as M
from screendata import make_screen
from matplotlib.lines import Line2D
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
NUM={}; CHK={}
J=lambda f: json.load(open('verify/'+f))
GREYB='#B8BEC8'
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def tc(ax,x,y,s,**k): ax.text(x,y,s,fontsize=k.pop('fs',7.8),color=k.pop('color',NAVY),**k)

# ======== 57.0-1 opener: accuracy against recall, and the ROC curve of the screen
s,y=make_screen(2000,0.05,2.0,0)
rules={'nobody ill':np.zeros_like(y),'screen, score > 1.5':(s>1.5).astype(int)}
fig,axs=plt.subplots(1,2,figsize=(4.15,1.9),dpi=240,gridspec_kw={'width_ratios':[1.05,0.95]})
ax=axs[0]; names=['accuracy','recall','precision']; w=0.36
for j,(k,p) in enumerate(rules.items()):
    v=[M.accuracy_score(y,p),M.recall_score(y,p),M.precision_score(y,p,zero_division=0)]
    NUM['rule|'+k]=v
    ax.bar(np.arange(3)+(j-0.5)*w,v,w*0.94,color=[OR,GREEN][j],hatch=['//',''][j],edgecolor=NAVY,lw=0.6,label=k)
    for x_,v_ in zip(np.arange(3)+(j-0.5)*w,v): ax.text(x_+(0.03 if j==0 else -0.03)*(-1),v_+0.02,'%.2f'%v_,ha=['right','left'][j],fontsize=7.8,color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=7.8); ax.set_ylim(0,1.6); ax.set_yticks([0,0.5,1]); ax.set_ylabel('score')
ax.legend(loc='upper center',fontsize=7.6,handlelength=1.1,labelspacing=0.2,borderaxespad=0.0,framealpha=1,edgecolor=GRID); grid(ax)
ax=axs[1]; f,t_,th=M.roc_curve(y,s,drop_intermediate=True)
ax.plot(f,t_,color=GREEN,lw=1.3); ax.plot([0,1],[0,1],color=GREY,lw=0.7,ls=':')
for thr,mk,off in ((0.5,'s',(0.05,-0.08)),(1.5,'o',(0.07,-0.12)),(2.5,'^',(0.07,-0.1))):
    i=np.argmin(abs(th[1:]-thr))+1; ax.plot(f[i],t_[i],mk,color=PUR,ms=4.2,mec=NAVY,mew=0.5); ax.text(f[i]+off[0],t_[i]+off[1],'t = %.1f'%thr,fontsize=7.8,color=PUR)
ax.plot([0],[0],'x',color=OR,ms=5,mew=1.2,clip_on=False); ax.text(0.07,0.01,'nobody ill',fontsize=7.8,color=OR)
ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate'); ax.set_xlim(-0.02,1); ax.set_ylim(0,1.02); grid(ax)
ax.text(0.97,0.28,'AUC %.3f'%M.roc_auc_score(y,s),ha='right',fontsize=7.8,color=NAVY)
fig.tight_layout(w_pad=0.8); sv(fig,'f570.png',400)

# ======== 57.2-1 the four cells and the scores that read them
fig,ax=plt.subplots(figsize=(4.55,2.15),dpi=240); ax.axis('off'); ax.set_xlim(0,10); ax.set_ylim(-1.3,4.1)
cells=[('TP','7',0,2,LB,BLUE),('FN','3',1,2,PINK,RED),('FP','19',0,1,PINK,RED),('TN','171',1,1,LB,BLUE)]
for lab,v,cx,cy,fc,ec in cells:
    ax.add_patch(Rectangle((2.6+cx*1.9,cy-0.0),1.9,1,fc=fc,ec=ec,lw=1.0)); ax.text(2.6+cx*1.9+0.95,cy+0.5,lab+'  '+v,ha='center',va='center',fontsize=9,color=NAVY,weight='bold')
ax.text(3.55,3.12,'flagged',ha='center',fontsize=7.8,color=NAVY); ax.text(5.45,3.12,'not flagged',ha='center',fontsize=7.8,color=NAVY)
ax.text(4.5,3.65,'screen result',ha='center',fontsize=8,color=GREY)
ax.text(2.45,2.5,'ill (10)',ha='right',va='center',fontsize=7.8,color=NAVY); ax.text(2.45,1.5,'healthy (190)',ha='right',va='center',fontsize=7.8,color=NAVY)
ax.text(0.0,3.65,'truth',ha='left',fontsize=8,color=GREY)
tc(ax,6.65,2.5,'recall = 7/10 = 0.70',va='center',color=GREEN)
tc(ax,6.65,1.5,'specificity = 171/190 = 0.90',va='center',color=GREEN)
tc(ax,3.55,0.55,'precision\n7/26 = 0.27',ha='center',va='center',color=PUR); tc(ax,5.45,0.55,'NPV\n171/174 = 0.98',ha='center',va='center',color=PUR)
tc(ax,2.6,-0.35,'accuracy = (7 + 171)/200 = 0.89  (the blue cells)',color=BLUE)
tc(ax,2.6,-0.95,'"nobody ill": TP = FP = 0, accuracy = 190/200 = 0.95',color=RED)
sv(fig,'f572.png',440)

# ======== 57.3-1 densities and the ROC curve (binormal screen)
fig,axs=plt.subplots(1,2,figsize=(4.6,1.95),dpi=240,gridspec_kw={'width_ratios':[1.1,0.9]})
ax=axs[0]; x=np.linspace(-3.2,5.2,400); t=1.5
ax.plot(x,norm.pdf(x,0,1),color=BLUE,lw=1.2); ax.plot(x,norm.pdf(x,2,1),color=OR,lw=1.2)
xm=x[x>=t]; ax.fill_between(xm,norm.pdf(xm,2,1),color=OR,alpha=0.35,hatch='//',edgecolor=OR,lw=0); ax.fill_between(xm,norm.pdf(xm,0,1),color=BLUE,alpha=0.45,lw=0)
ax.axvline(t,color=PUR,lw=1.1,ls='--'); tc(ax,t-0.08,0.43,'t = 1.5',color=PUR,ha='right')
tc(ax,-2.9,0.36,'healthy',color=BLUE); tc(ax,3.55,0.39,'ill',color=OR)
tc(ax,2.9,0.2,'TPR 0.691',color=OR,ha='left',bbox=dict(fc='white',ec='none',pad=0.5,alpha=0.85)); tc(ax,2.1,0.045,'FPR 0.067',color=BLUE,ha='left',bbox=dict(fc='white',ec='none',pad=0.5,alpha=0.85))
ax.set_xlabel('score'); ax.set_ylabel('density'); ax.set_ylim(0,0.5); ax.set_xlim(-3.2,5.2); grid(ax)
ax=axs[1]; tt=np.linspace(-6,8,600); fpr_=norm.sf(tt); tpr_=norm.sf(tt-2)
ax.fill_between(fpr_,tpr_,color=GREEN,alpha=0.18,lw=0); ax.plot(fpr_,tpr_,color=GREEN,lw=1.3); ax.plot([0,1],[0,1],color=GREY,lw=0.7,ls=':')
ax.plot(norm.sf(t),norm.sf(t-2),'o',color=PUR,ms=4.5,mec=NAVY,mew=0.5); tc(ax,norm.sf(t)+0.06,norm.sf(t-2)-0.13,'t = 1.5',color=PUR)
tc(ax,0.97,0.22,'AUC = 0.921',ha='right'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_xlim(0,1); ax.set_ylim(0,1.02); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f573.png',440)

# ======== 57.5-1 by hand: ROC staircase and precision-recall steps of eight scores
yb=np.array([1,0,1,1,0,1,0,0]); sb=np.array([.9,.8,.7,.6,.6,.4,.3,.2])
fpr,tpr,thr=M.roc_curve(yb,sb,drop_intermediate=False); pr,rc,_=M.precision_recall_curve(yb,sb)
fig,axs=plt.subplots(1,2,figsize=(4.6,2.0),dpi=240)
ax=axs[0]; ax.fill_between(fpr,tpr,color=GREEN,alpha=0.18,lw=0); ax.plot(fpr,tpr,'-o',color=GREEN,lw=1.3,ms=3.5)
for f_,t_,h in zip(fpr[1:],tpr[1:],thr[1:]): ax.text(f_+0.03,t_-0.1 if f_<0.9 else t_-0.1,'%.1f'%h,fontsize=7.8,color=NAVY)
ax.plot([0,1],[0,1],color=GREY,lw=0.7,ls=':'); tc(ax,0.97,0.12,'AUC = 0.719',ha='right'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_xlim(-0.02,1.02); ax.set_ylim(0,1.05); grid(ax)
ax=axs[1]
Rj=[.25,.25,.5,.75,1.0]; Pj=[1,.5,2/3,.6,2/3]; Rprev=[0]+Rj[:-1]   # Eq. 57.3-9: width R_j - R_(j-1), height P_j
for a_,b_,p_ in zip(Rprev,Rj,Pj):
    if b_>a_: ax.add_patch(Rectangle((a_,0),b_-a_,p_,fc=RED,alpha=0.18,ec=RED,lw=0.8))
ax.plot(rc,pr,'o--',color=RED,lw=0.9,ms=3.2)
tc(ax,0.04,0.14,'AP = 0.733 (steps)',color=RED); tc(ax,0.04,0.03,'dashed, trapezoids: 0.713',color=GREY)
ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(-0.02,1.02); ax.set_ylim(0,1.1); grid(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f575.png',440)

# ======== 57.6-1 how a scorer is made
fig,ax=new(4.6,1.35); ax.set_xlim(0,12.2); ax.set_ylim(-1.1,3.0)
labs=[('estimator,\nX, y',0.0),('response\nmethod',2.55),('score\nfunction',5.1),('sign',7.65),('scorer\nvalue',9.8)]
wd=[2.1,2.1,2.1,1.6,2.1]
for (l,x0),w_ in zip(labs,wd): box(ax,x0,1.45,w_,1.35,l,fc=LB,ec=BLUE,fs=8.4)
for a,b_ in ((2.1,2.55),(4.65,5.1),(7.2,7.65),(9.25,9.8)): arr(ax,(a,2.12),(b_,2.12),c=NAVY)
tc(ax,0.0,0.82,'response method: predict, predict_proba or decision_function',color=PUR,fs=7.8,va='top')
tc(ax,0.0,0.18,'score function: f(y_true, ·, **kwargs), e.g. fbeta_score with beta=2',color=GREEN,fs=7.8,va='top')
tc(ax,0.0,-0.46,'sign: +1, or −1 when greater_is_better=False',color=RED,fs=7.8,va='top')
sv(fig,'f576.png',440)

# ======== 57.3-2 losses for the true class, and which forecast minimises the expected loss
fig,axs=plt.subplots(1,2,figsize=(4.6,1.95),dpi=240)
ax=axs[0]; p=np.linspace(0.01,1,300)
ax.plot(p,-np.log(p),color=RED,lw=1.3,label='log loss'); ax.plot(p,(1-p)**2,color=GREEN,lw=1.3,ls='--',label='Brier'); ax.step([0,0.5,0.5,1],[1,1,0,0],color=GREY,lw=1.1,ls=':',label='0-1 loss')
ax.set_ylim(0,3.1); ax.set_xlim(0,1); ax.set_xlabel('probability given to the true class'); ax.set_ylabel('loss'); ax.legend(loc='upper right',fontsize=7.8,handlelength=1.8,labelspacing=0.2,borderaxespad=0.1); grid(ax)
ax=axs[1]; q=np.linspace(0.03,0.97,300); pi=0.7
ax.plot(q,-(pi*np.log(q)+(1-pi)*np.log(1-q)),color=RED,lw=1.3); ax.plot(q,(pi-q)**2+pi*(1-pi),color=GREEN,lw=1.3,ls='--')
ax.axvline(pi,color=PUR,lw=1,ls=':'); tc(ax,pi+0.02,0.07,'q = π',color=PUR)
tc(ax,0.80,0.92,'log loss',color=RED); tc(ax,0.04,0.33,'Brier',color=GREEN)
ax.set_ylim(0,1.3); ax.set_xlim(0,1); ax.set_xlabel('forecast q  (true probability π = 0.7)'); ax.set_ylabel('expected loss'); grid(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f577.png',440)

# ======== 57.4-1 regression losses and deviances
fig,axs=plt.subplots(1,2,figsize=(4.6,1.95),dpi=240)
ax=axs[0]; r=np.linspace(-3,3,400)
ax.plot(r,np.abs(r),color=BLUE,lw=1.3,label='|r|'); ax.plot(r,r**2,color=RED,lw=1.3,ls='--',label='r²'); ax.plot(r,np.maximum(0.8*r,-0.2*r),color=GREEN,lw=1.3,ls='-.',label='pinball, α = 0.8')
ax.set_ylim(0,5); ax.set_xlim(-3,3); ax.set_xlabel('residual r = y − ŷ'); ax.set_ylabel('loss of one sample'); ax.legend(loc='upper center',fontsize=7.8,handlelength=1.8,labelspacing=0.2,borderaxespad=0.1,ncol=1); grid(ax)
ax=axs[1]; mu=np.exp(np.linspace(np.log(0.7),np.log(14),300)); yv=4.0
from sklearn.metrics import mean_tweedie_deviance as mtd
for pw,col,ls,lab in ((0,BLUE,'-','q = 0, squared'),(1,RED,'--','q = 1, Poisson'),(2,GREEN,'-.','q = 2, gamma')):
    d=np.array([mtd([yv],[m_],power=pw) for m_ in mu]); d1=mtd([yv],[1.0],power=pw)
    ax.plot(mu,d/d1,color=col,ls=ls,lw=1.3,label=lab)
ax.axvline(yv,color=GREY,lw=0.8,ls=':'); tc(ax,yv*1.06,2.2,'y = 4',color=GREY); ax.set_xscale('log'); ax.set_xticks([1,2,4,8]); ax.set_xticklabels(['1','2','4','8'])
ax.set_ylim(0,4.4); ax.set_xlabel('prediction μ'); ax.set_ylabel('deviance ÷ value at μ = 1'); ax.legend(loc='upper left',fontsize=7.8,handlelength=1.8,labelspacing=0.2,borderaxespad=0.1); grid(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f578.png',440)

# ======== 57.9-1 prevalence: precision-recall curves and scores against prevalence
d=2.0; tpr15=norm.sf(1.5-d); fpr15=norm.sf(1.5); prevs=[0.5,0.1,0.01]
fig,axs=plt.subplots(1,2,figsize=(4.7,2.1),dpi=240,gridspec_kw={'width_ratios':[0.9,1.1]})
ax=axs[0]; cols=[BLUE,OR,RED]; lss=['-','--','-.']; big={}
for pi,col,ls in zip(prevs,cols,lss):
    sb_,yb_=make_screen(200000,pi,d,int(pi*1000)); p_,r_,_=M.precision_recall_curve(yb_,sb_)
    ax.plot(r_,p_,color=col,ls=ls,lw=1.3,label='π = %g'%pi); ax.axhline(pi,color=col,lw=0.6,ls=':')
    big[pi]=(M.roc_auc_score(yb_,sb_),M.average_precision_score(yb_,sb_))
NUM['prev_auc_ap']={str(k):v for k,v in big.items()}
ax.set_xlabel('recall'); ax.set_ylabel('precision'); ax.set_xlim(0,1); ax.set_ylim(0,1.02); ax.legend(loc='lower left',fontsize=7.8,handlelength=1.8,labelspacing=0.2,borderaxespad=0.1,framealpha=1,edgecolor=GRID); grid(ax)
ax=axs[1]; pis=np.logspace(-3,np.log10(0.5),14); rows=[]
for pi in pis:
    sb_,yb_=make_screen(100000,pi,d,7); yh=(sb_>1.5).astype(int)
    rows.append([M.roc_auc_score(yb_,sb_),M.average_precision_score(yb_,sb_),M.accuracy_score(yb_,yh),M.f1_score(yb_,yh),M.matthews_corrcoef(yb_,yh),M.balanced_accuracy_score(yb_,yh)])
rows=np.array(rows); NUM['prev_grid']={'pi':pis.tolist(),'auc,ap,acc,f1,mcc,bal':rows.tolist()}
for j,(lab,col,ls) in enumerate((('ROC AUC',GREEN,'-'),('accuracy',GREY,'--'),('MCC',PUR,'-.'),('F1',OR,'-'),('AP',RED,'--'))):
    idx=[0,2,4,3,1][j]; ax.plot(pis,rows[:,idx],color=col,ls=ls,lw=1.3,label=lab)
ax.set_xscale('log'); ax.set_xlabel('prevalence π'); ax.set_ylabel('score at t = 1.5'); ax.set_ylim(0,1.02); ax.set_xlim(0.001,0.5)
ax.legend(loc='center left',bbox_to_anchor=(0.0,0.62),fontsize=7.8,handlelength=1.6,labelspacing=0.2,borderaxespad=0.1,framealpha=1,edgecolor=GRID)
ax.set_yticks([0,0.5,1]); grid(ax); fig.tight_layout(w_pad=0.9); sv(fig,'f579.png',450)

# ======== 57.9-2 F-beta against the threshold, and the harmonic mean
fig,axs=plt.subplots(1,2,figsize=(4.7,2.0),dpi=240,gridspec_kw={'width_ratios':[1.15,0.85]})
ax=axs[0]; ts=np.linspace(-0.5,4,300); pi=0.05; tpr_=norm.sf(ts-2); fpr_=norm.sf(ts)
def fb(b): return (1+b**2)*tpr_*pi/((1+b**2)*tpr_*pi+b**2*(1-tpr_)*pi+fpr_*(1-pi))
best={}
for b,col,ls in ((0.5,BLUE,'-'),(1,GREEN,'--'),(2,RED,'-.')):
    f_=fb(b); ax.plot(ts,f_,color=col,ls=ls,lw=1.3); k=np.argmax(f_); ax.plot(ts[k],f_[k],'o',color=col,ms=4,mec=NAVY,mew=0.5); best[b]=(float(ts[k]),float(f_[k]))
acc=pi*tpr_+(1-pi)*(1-fpr_); ax.plot(ts,acc,color=GREY,lw=1.1,ls=':')
NUM['fbeta_best']=best; NUM['acc_at_best']={'acc_max_t':float(ts[np.argmax(acc)]),'acc_max':float(acc.max())}

ax.set_xlabel('threshold t (π = 0.05)'); ax.set_ylabel('score'); ax.set_ylim(0,1.3); ax.set_yticks([0,0.5,1]); ax.set_xlim(-0.5,4); grid(ax)
ax.legend(handles=[Line2D([],[],color=c_,ls=l_,lw=1.3) for c_,l_ in ((BLUE,'-'),(GREEN,'--'),(RED,'-.'),(GREY,':'))],labels=['F0.5','F1','F2','accuracy'],loc='upper center',ncol=4,fontsize=7.8,handlelength=1.5,columnspacing=0.8,handletextpad=0.4,borderaxespad=0.1,frameon=False)
ax=axs[1]; P=np.linspace(0.01,1,300); R_=0.9
ax.plot(P,(P+R_)/2,color=GREY,ls=':',lw=1.2); ax.plot(P,2*P*R_/(P+R_),color=GREEN,lw=1.3); ax.plot(P,np.minimum(P,R_),color=BLUE,ls='--',lw=1.0)
tc(ax,0.5,0.76,'mean',color=GREY,rotation=0,ha='right'); tc(ax,0.55,0.50,'F1',color=GREEN); tc(ax,0.28,0.08,'min',color=BLUE)
ax.set_xlabel('precision P  (recall R = 0.9)'); ax.set_ylabel('average of P and R'); ax.set_xlim(0,1); ax.set_ylim(0,1.02); grid(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f5710.png',450)

# ======== 57.9-3 regression: one outlier, and residuals of OLS on diabetes
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
rng=np.random.RandomState(0); yt=rng.normal(100,15,60); yp=yt+rng.normal(0,5,60)
dl=np.linspace(0,100,26); res={'MAE':[],'RMSE':[],'MedAE':[],'1-R2':[]}
for dd in dl:
    q=yp.copy(); q[0]+=dd
    res['MAE'].append(M.mean_absolute_error(yt,q)); res['RMSE'].append(M.root_mean_squared_error(yt,q)); res['MedAE'].append(M.median_absolute_error(yt,q)); res['1-R2'].append(1-M.r2_score(yt,q))
NUM['outlier']={k:[v[0],v[-1]] for k,v in res.items()}; NUM['outlier_dl']=[float(dl[0]),float(dl[-1])]
fig,axs=plt.subplots(1,2,figsize=(4.7,2.05),dpi=240,gridspec_kw={'width_ratios':[1,1]})
ax=axs[0]
for k,col,ls in (('MAE',BLUE,'-'),('RMSE',RED,'--'),('MedAE',GREEN,'-.'),('1-R2',PUR,':')):
    v=np.array(res[k]); ax.plot(dl,v/v[0],color=col,ls=ls,lw=1.3)
ax.set_yscale('log'); ax.set_xlabel('error added to one prediction'); ax.set_ylabel('score ÷ score at 0'); ax.set_ylim(0.8,12); ax.set_yticks([1,3,10]); ax.set_yticklabels(['1','3','10'])
for k,col,dy,va in (('1-R2',PUR,1.15,'bottom'),('RMSE',RED,1.15,'bottom'),('MAE',BLUE,1.1,'bottom'),('MedAE',GREEN,0.92,'top')):
    v=np.array(res[k])/res[k][0]; j=18; ax.text(dl[j],v[j]*dy,k.replace('1-R2','1 − R²'),color=col,fontsize=7.8,va=va,ha='center')
grid(ax)
Xd,yd=load_diabetes(return_X_y=True); Xa,Xb,ya,yb=train_test_split(Xd,yd,test_size=0.25,random_state=0)
pd_=LinearRegression().fit(Xa,ya).predict(Xb)
from sklearn.metrics import PredictionErrorDisplay as PED
ax=axs[1]; PED.from_predictions(yb,pd_,kind='residual_vs_predicted',ax=ax,scatter_kwargs=dict(s=7,color=BLUE,alpha=0.7,linewidths=0),line_kwargs=dict(color=RED,lw=1.0))
ax.set_xlabel('predicted value ŷ'); ax.set_ylabel('residual y − ŷ'); grid(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f5711.png',450)

# ======== 57.11-1 case study: ROC (zoomed) and PR curves of the models
cc=J('case_cancer.json'); yc=np.array(cc['y'])
order=[('logistic',GREEN,'-'),('SVC',BLUE,'--'),('GB',OR,'-.'),('HGB',PUR,':'),('forest',RED,'-'),('naive Bayes',GREY,'--')]
fig,axs=plt.subplots(1,2,figsize=(4.7,2.35),dpi=240)
for nm,col,ls in order:
    sc_=np.array(cc['models'][nm]['score']); f,t_,_=M.roc_curve(yc,sc_); p_,r_,_=M.precision_recall_curve(yc,sc_)
    axs[0].plot(f,t_,color=col,ls=ls,lw=1.25,label=nm); axs[1].plot(r_,p_,color=col,ls=ls,lw=1.25)
axs[0].set_xlim(0,0.3); axs[0].set_ylim(0.6,1.01); axs[0].set_xlabel('FPR (zoomed to 0–0.3)'); axs[0].set_ylabel('TPR (malignant)'); grid(axs[0])
axs[1].set_xlim(0.5,1.0); axs[1].set_ylim(0.6,1.01); axs[1].set_xlabel('recall (zoomed to 0.5–1)'); axs[1].set_ylabel('precision'); grid(axs[1])
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=6,fontsize=7.8,handlelength=1.6,columnspacing=0.8,handletextpad=0.4,frameon=False)
fig.tight_layout(rect=(0,0.08,1,1),w_pad=0.9); sv(fig,'f5712.png',450)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
for k,v in CHK.items(): print(k,v)

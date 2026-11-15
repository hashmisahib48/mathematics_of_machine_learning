# All figures of Chapter 53. Run: python3 fig/figs.py (numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectFromModel, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
Xa,Xb,ya,yb,names=load(); names=[str(a) for a in names]; Z=StandardScaler().fit_transform(Xa)
lr=lambda: LogisticRegression(max_iter=1000)
F,P=f_classif(Xa,ya); orderF=np.argsort(-F); rankF=np.empty(30,int); rankF[orderF]=np.arange(30)
l1=lambda C: LogisticRegression(l1_ratio=1,C=C,solver='liblinear',random_state=0)
sel_f=SelectKBest(k=5).fit(Z,ya).get_support(); sel_r=RFE(lr(),n_features_to_select=5).fit(Z,ya).get_support(); sel_l=SelectFromModel(l1(0.05)).fit(Z,ya).get_support()
NUM['sets']={'filter':np.flatnonzero(sel_f).tolist(),'rfe':np.flatnonzero(sel_r).tolist(),'l1':np.flatnonzero(sel_l).tolist(),'overlap_all3':int((sel_f&sel_r&sel_l).sum())}

# ======== 53.0-1 opener
fig,axs=plt.subplots(2,1,figsize=(4.4,2.3),dpi=240,sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
ax=axs[0]; ax.bar(np.arange(30),F[orderF],color=BLUE,width=0.75); ax.set_yscale('log'); ax.set_ylim(5e-4,3e3); ax.set_ylabel('ANOVA $F$'); grid(ax)
ax.set_title('30 features in the order of their $F$ score',fontsize=9,color=NAVY)
ax=axs[1]
for row,(m,col,mk,lab) in enumerate(((sel_l,PUR,'D','embedded: L1'),(sel_r,GREEN,'s','wrapper: RFE'),(sel_f,OR,'o','filter: top $F$'))):
    idx=np.flatnonzero(m); ax.plot(rankF[idx],np.full(len(idx),row),mk,color=col,ms=6)
    ax.text(30.2,row,lab,fontsize=8.5,va='center',color=col)
ax.set_ylim(-0.7,2.7); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-0.8,30); ax.set_xticks([0,5,10,15,20,25,29]); ax.set_xticklabels([1,6,11,16,21,26,30]); ax.set_xlabel('position of the feature in the order of $F$')
fig.tight_layout(h_pad=0.4); fig.subplots_adjust(right=0.74); sv(fig,'f530.png',420)

# ======== 53.1-1 correlation heat map
R=np.abs(np.corrcoef(Xa,rowvar=False)); Lk=linkage(squareform(1-R,checks=False),'average'); o=leaves_list(Lk)
fig,ax=plt.subplots(figsize=(2.98,2.65),dpi=240)
im=ax.imshow(R[np.ix_(o,o)],cmap='Blues',vmin=0,vmax=1)
ax.set_xticks(range(0,30,2)); ax.set_yticks(range(0,30,2)); ax.set_xticklabels([rankF[o[j]]+1 for j in range(0,30,2)],fontsize=8.2); ax.set_yticklabels([rankF[o[j]]+1 for j in range(0,30,2)],fontsize=8.2); ax.tick_params(length=1.5,pad=1)
ax.spines[:].set_visible(False)
for t in (ax.get_xticklabels()):
    pass
top=[k for k,j in enumerate(o) if rankF[j]<5]
for k in top: ax.plot(k,-1.4,'v',color=OR,ms=4.5,clip_on=False)
cb=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.02); cb.set_label('$|r|$',fontsize=9); cb.ax.tick_params(labelsize=8)
ax.set_xlabel('feature (position in the order of $F$), grouped by correlation',fontsize=8.5)
NUM['corr']={'top5_positions_in_order':top,'order_rank':[int(rankF[j]+1) for j in o]}
sv(fig,'f531.png',270)

# ======== 53.3-2 multiple testing
rng=np.random.RandomState(0); y=np.arange(100)%2; Xs=rng.standard_normal((100,200)); Xs[:,:20]+=0.6*y[:,None]
Fs,Ps=f_classif(Xs,y); o=np.argsort(Ps); m=200; al=0.05
fig,axs=plt.subplots(1,2,figsize=(4.4,1.75),dpi=240,gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]; k=np.arange(1,m+1); true=(o<20)
ax.plot(k[true],Ps[o][true],'o',color=BLUE,ms=3.2,label='shifted'); ax.plot(k[~true],Ps[o][~true],'s',mfc='white',mec=GREY,ms=3.2,mew=0.8,label='unrelated')
ax.axhline(al,color=RED,lw=1.1,ls='-'); ax.plot(k,al*k/m,color=GREEN,lw=1.4,ls='--'); ax.axhline(al/m,color=PUR,lw=1.2,ls=':')
ax.text(41,al+0.003,'$\\alpha$',color=RED,fontsize=8.5,ha='right',va='bottom'); ax.text(41,al*40/m+0.0045,'$\\alpha k/m$',color=GREEN,fontsize=8.5,ha='right',va='bottom'); ax.text(41,-0.0058,'$\\alpha/m$',color=PUR,fontsize=8.5,ha='right',va='bottom')
ax.set_xlim(0,41); ax.set_ylim(-0.006,0.09); ax.set_xlabel('rank $k$ of the $p$-value'); ax.set_ylabel('$p_{(k)}$'); ax.legend(loc='upper left',fontsize=8,handletextpad=0.2,borderaxespad=0.1,labelspacing=0.15); grid(ax)
ks={'Fpr':(SelectFpr(alpha=al).fit(Xs,y).get_support()),'Fdr':SelectFdr(alpha=al).fit(Xs,y).get_support(),'Fwe':SelectFwe(alpha=al).fit(Xs,y).get_support()}
NUM['mt_one_run']={n:[int(s[:20].sum()),int(s[20:].sum())] for n,s in ks.items()}
pn=json.load(open('verify/prose_numbers.json'))['multitest']
ax=axs[1]; w=0.36
for i,(nm_,key) in enumerate((('Fpr','fpr'),('Fdr','fdr'),('Fwe','fwe'))):
    t,f_,_=pn[key]; ax.bar(i-w/2,t,w,color=BLUE); ax.bar(i+w/2,f_,w,color=OR,hatch='///',edgecolor='white',lw=0)
    ax.text(i-w/2,t+0.4,'%.1f'%t,ha='center',fontsize=8.2); ax.text(i+w/2,f_+0.4,'%.1f'%f_,ha='center',fontsize=8.2)
ax.set_xticks(range(3)); ax.set_xticklabels(['Fpr','Fdr','Fwe']); ax.set_ylim(0,21.5); ax.set_ylabel('features kept (mean of 200 runs)',fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=BLUE,label='shifted (20)'),Patch(fc=OR,hatch='///',ec='white',label='unrelated (180)')],loc='upper right',fontsize=7.8,handlelength=1,handletextpad=0.4,borderaxespad=0.1); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f534.png',400)

# ======== 53.5-1 selection bias
n,k_=60,10; ps=[10,30,100,300,1000]; leak={p:[] for p in ps}; hon={p:[] for p in ps}; reps=24
for seed in range(reps):
    rng=np.random.RandomState(seed); yy=np.arange(n)%2; cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    for p in ps:
        X=rng.standard_normal((n,p)); kk=min(k_,p)
        leak[p].append(cross_val_score(lr(),SelectKBest(k=kk).fit_transform(X,yy),yy,cv=cv).mean()); hon[p].append(cross_val_score(make_pipeline(SelectKBest(k=kk),lr()),X,yy,cv=cv).mean())
NUM['bias']={p:[float(np.mean(leak[p])),float(np.mean(hon[p]))] for p in ps}
ms_=[10,30,100,300,1000,3000]; sim=[]
for m_ in ms_:
    r=[]
    for s in range(40):
        rng=np.random.RandomState(100+s); X=rng.standard_normal((n,m_)); yy=rng.standard_normal(n)
        Xc=(X-X.mean(0))/X.std(0); yc=(yy-yy.mean())/yy.std(); r.append(np.abs(Xc.T@yc/n).max())
    sim.append(np.mean(r))
NUM['maxr']={'m':ms_,'sim':[float(v) for v in sim],'formula':[float(np.sqrt(2*np.log(m_)/n)) for m_ in ms_]}
fig,ax=plt.subplots(figsize=(3.2,1.7),dpi=240)
ax.plot(ps,[np.mean(leak[p]) for p in ps],'-o',color=RED,ms=3.5,lw=1.3,label='selected before CV'); ax.plot(ps,[np.mean(hon[p]) for p in ps],'-s',color=GREEN,ms=3.5,lw=1.3,label='selected inside CV')
ax.axhline(0.5,color=GREY,lw=0.9,ls=':'); ax.set_xscale('log'); ax.set_xlabel('number of unrelated features $m$'); ax.set_ylabel('CV accuracy'); ax.set_ylim(0.4,1.0); ax.legend(loc='upper left',fontsize=8,handlelength=1.4,borderaxespad=0.1,labelspacing=0.15); grid(ax)
fig.tight_layout(); sv(fig,'f535.png',310)

# ======== 53.7-1 algorithm: RFE and forward selection loops
fig,ax=plt.subplots(figsize=(4.4,1.95),dpi=240); ax.set_xlim(-0.8,12.8); ax.set_ylim(-0.2,6.4); ax.axis('off')
ax.text(2.7,5.95,'RFE (backward)',fontsize=9.5,weight='bold',color=GREEN,ha='center',va='center'); ax.text(9.0,5.95,'forward selection',fontsize=9.5,weight='bold',color=GREEN,ha='center',va='center')
L=[(0.2,4.4,5.0,0.9,'$S$ = all $p$ features',LB,BLUE),(0.2,2.8,5.0,1.3,'fit the model on $S$,\nrank by importance $c_j$',LGREY,GREY),(0.2,1.2,5.0,1.3,'drop the lowest\nmin(step, $|S|-s$)',PINK,RED),(0.2,0.0,5.0,0.9,'$|S|=s$?  yes: stop',"#E3F4EA",GREEN)]
for x,y,w,h,t,fc,ec in L: box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=8.2)
for y0,y1 in ((4.4,4.1),(2.8,2.5),(1.2,0.9)): arr(ax,(2.7,y0),(2.7,y1),c=NAVY,lw=1,ms=7)
ax.annotate('',xy=(0.2,3.45),xytext=(0.2,1.85),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1,connectionstyle='arc3,rad=0.5')); 
R2=[(6.4,4.4,5.2,0.9,'$S=\\emptyset$',LB,BLUE),(6.4,2.8,5.2,1.3,'score $S\\cup\\{j\\}$ by CV\nfor every $j\\notin S$',LGREY,GREY),(6.4,1.2,5.2,1.3,'add the best $j$\n(gain $\\geq$ tol, if set)',PINK,RED),(6.4,0.0,5.2,0.9,'$|S|=s$?  yes: stop',"#E3F4EA",GREEN)]
for x,y,w,h,t,fc,ec in R2: box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=8.2)
for y0,y1 in ((4.4,4.1),(2.8,2.5),(1.2,0.9)): arr(ax,(9.0,y0),(9.0,y1),c=NAVY,lw=1,ms=7)
ax.annotate('',xy=(11.6,3.45),xytext=(11.6,1.85),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1,connectionstyle='arc3,rad=-0.5'))
sv(fig,'f536.png',400)

# ======== 53.10-1 gallery: L1 path, bars of |w|
fig,axs=plt.subplots(1,4,figsize=(4.4,1.4),dpi=240,sharey=True)
for ax,C in zip(axs,(0.02,0.05,0.2,1.0)):
    w=np.abs(l1(C).fit(Z,ya).coef_[0]); keep=w>=1e-5; wo=w[orderF]; ko=keep[orderF]
    ax.bar(np.arange(30),wo,color=[GREEN if q else LGREY for q in ko],width=0.8); ax.set_title('$C=%g$: %d kept'%(C,keep.sum()),fontsize=8.5,color=NAVY); ax.set_xticks([0,10,20,29]); ax.set_xticklabels([1,11,21,30],fontsize=8)
    grid(ax)
axs[0].set_ylabel('$|w_j|$')
fig.tight_layout(w_pad=0.4,rect=(0,0.05,1,1)); fig.text(0.5,0.0,'feature position in the order of $F$',fontsize=8.5,ha='center'); sv(fig,'f537.png',390)

# ======== RFECV numbers quoted in 53.10 (no figure)
rc=RFECV(lr(),cv=StratifiedKFold(5,shuffle=True,random_state=0),min_features_to_select=1).fit(Z,ya)  # (scaled on training rows; the picture only shows the curve)
cr=rc.cv_results_; nf=cr['n_features']; mu=cr['mean_test_score']; sd=cr['std_test_score']
kb=[np.mean(cross_val_score(make_pipeline(SelectKBest(k=k),lr()),Z,ya,cv=StratifiedKFold(5,shuffle=True,random_state=0))) for k in range(1,31)]
NUM['rfecv']={'n_features_':int(rc.n_features_),'best_mean':float(mu.max()),'at_best':int(nf[mu.argmax()]),'curve_first':[float(mu[0]),float(mu[4])],'kbest_curve':[float(v) for v in kb]}

# ======== 53.11-1 case study
res=json.load(open('verify/case_bc.json')); ks=res['k']
fig,axs=plt.subplots(1,2,figsize=(4.4,1.8),dpi=240,sharey=True)
for ax,(idx,ttl) in zip(axs,((0,'5-fold CV on the 426 training rows'),(1,'test accuracy (143 rows)'))):
    ax.plot(ks,[v[idx] for v in res['filter']],'-o',color=OR,ms=3,lw=1.2,label='filter: top $F$'); ax.plot(ks,[v[idx] for v in res['rfe']],'-s',color=GREEN,ms=3,lw=1.2,label='wrapper: RFE')
    ax.plot([v[1] for v in res['l1']],[v[2+idx] for v in res['l1']],'D',color=PUR,ms=3.6,label='embedded: L1')
    ax.set_title(ttl,fontsize=8.5,color=NAVY); ax.set_xlabel('number of features kept'); grid(ax); ax.set_xlim(0,31)
axs[0].set_ylabel('accuracy'); axs[0].set_ylim(0.86,1.0); axs[0].legend(loc='lower right',fontsize=7.8,handlelength=1.3,borderaxespad=0.2,labelspacing=0.15)
fig.tight_layout(w_pad=0.6); sv(fig,'f539.png',390)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1,default=str); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})

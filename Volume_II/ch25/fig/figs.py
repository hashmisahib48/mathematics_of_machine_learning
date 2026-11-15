# All figures of Chapter 25. Run from anywhere: python3 fig/figs.py
# (PNGs into ch25/, numbers into verify/fig_numbers.json)
import sys,os,json,math,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestClassifier as RFC, ExtraTreesClassifier as ETC, BaggingClassifier as BAG
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
ONLY=set(sys.argv[1:])
def want(k): return (not ONLY) or (k in ONLY)
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM=json.load(open('verify/fig_numbers.json')) if os.path.exists('verify/fig_numbers.json') else {}
CHK=json.load(open('verify/fig_textsize.json')) if os.path.exists('verify/fig_textsize.json') else {}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
LO='#FDEBD0'
CMAP=ListedColormap([LB,LO])
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def cls_pts(ax,X,y,s=30,**k):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.7,zorder=5,**k)

# ---- the six-applicant toy (renumbered 1..6), as in code/toy.py
INC=np.array([18,22,36,40,55,62.]); REP=np.array([0,0,1,1,0,1]); DRAWS=[[0,2,4,4,5,5],[1,1,3,3,3,5],[0,0,0,1,2,3]]
stumps=[DTC(max_depth=1,random_state=0).fit(INC[:,None],REP,sample_weight=np.bincount(d,minlength=6)) for d in DRAWS]
XG=np.linspace(10,92,821)[:,None]
PS=[s.predict_proba(XG)[:,1] for s in stumps]; PAVG=np.mean(PS,axis=0)

# ---- 25.0-1 opener: the committee, schematic and toy
if want('f250'):
    fig,axs=plt.subplots(1,2,figsize=(5.3,2.45),dpi=240,gridspec_kw={'width_ratios':[1.12,1]})
    ax=axs[0]; ax.set_xlim(-0.1,6.2); ax.set_ylim(0,5.2); ax.axis('off')
    box(ax,0.0,2.0,1.15,1.2,'training\nset',fc=LGREY,ec=GREY,fs=8.6)
    ys=[3.85,2.2,0.55]; cols=[BLUE,GREEN,PUR]
    for k,yk in enumerate(ys):
        arr(ax,(1.15,2.6),(1.55,yk+0.35),c=GREY,lw=0.9,ms=7)
        box(ax,1.55,yk,1.3,0.7,f'sample {k+1}',fc=LB,ec=BLUE,fs=8.4)
        arr(ax,(2.85,yk+0.35),(3.2,yk+0.35),c=GREY,lw=0.9,ms=7)
        box(ax,3.2,yk,1.05,0.7,f'tree {k+1}',fc='#E3F4EA',ec=GREEN,fs=8.4)
        arr(ax,(4.25,yk+0.35),(4.65,2.6),c=GREY,lw=0.9,ms=7)
    box(ax,4.65,2.0,1.45,1.2,'average\nof trees',fc='#FDF0E0',ec=OR,fs=8.4)
    ax.text(2.2,4.85,'draw $n$ with\nreplacement',fontsize=8.4,color=GREY,ha='center',va='bottom')
    ax.text(3.72,4.85,'grow\neach tree',fontsize=8.4,color=GREY,ha='center',va='bottom'); ax.text(5.38,1.65,'(classes: vote\nor average $P$)',fontsize=8.0,color=GREY,ha='center',va='top')
    ax=axs[1]
    for p,ls,col,lab in zip(PS,['--','-.',':'],[BLUE,PUR,RED],['stump 1','stump 2','stump 3']): ax.plot(XG[:,0],p,color=col,lw=1.1,ls=ls,label=lab)
    ax.plot(XG[:,0],PAVG,color=GREEN,lw=2.2,label='average')
    for x_,y_ in zip(INC,REP): ax.scatter(x_,y_,s=34,color=[BLUE,OR][y_],marker='os'[y_],edgecolor='white',lw=0.7,zorder=6,clip_on=False)
    ax.set_xlim(10,92); ax.set_xticks([20,40,60,80]); ax.set_ylim(-0.08,1.08); ax.set_xlabel('income (thousands)'); ax.set_ylabel('$P$(repaid)'); ax.set_yticks([0,0.5,1]); grid(ax)
    ax.legend(loc='center right',bbox_to_anchor=(1.0,0.5),fontsize=8.2,borderaxespad=0.1,handlelength=1.8,labelspacing=0.3)
    fig.tight_layout(w_pad=0.6); sv(fig,'f250.png',470)

# ---- moons data as in Chapter 24
Xm,ym=make_moons(n_samples=300,noise=0.3,random_state=0); Xmt,Xme,ymt,yme=train_test_split(Xm,ym,test_size=0.33,random_state=0,stratify=ym)
gx,gy=np.meshgrid(np.linspace(-1.7,2.7,300),np.linspace(-1.5,1.8,300)); GRIDXY=np.c_[gx.ravel(),gy.ravel()]
from matplotlib.colors import LinearSegmentedColormap
PCM=LinearSegmentedColormap.from_list('p',['#B4CFEF','#ECEAE8','#F8D49A'])
def region(ax,Z,hard=True):
    if hard:
        ax.contourf(gx,gy,Z.reshape(gx.shape),levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx,gy,Z.reshape(gx.shape),levels=[0.5],colors=[GREEN],linewidths=1.3)
    else:
        ax.contourf(gx,gy,Z.reshape(gx.shape),levels=np.linspace(0,1,11),cmap=PCM); ax.contour(gx,gy,Z.reshape(gx.shape),levels=[0.5],colors=[GREEN],linewidths=1.5)

# ---- 25.2-1 three trees on bootstrap samples and their average
if want('f252'):
    bag=BAG(DTC(),n_estimators=100,random_state=0).fit(Xmt,ymt)
    accs=[float(t.score(Xme[:,bag.estimators_features_[k]],yme)) for k,t in enumerate(bag.estimators_)]
    NUM['moons']={'tree_acc_mean':round(float(np.mean(accs)),4),'tree_acc_min':round(min(accs),4),'tree_acc_max':round(max(accs),4),'bag_test':round(float(bag.score(Xme,yme)),4),
       'single_tree_test':round(float(DTC(random_state=0).fit(Xmt,ymt).score(Xme,yme)),4),'first3':[round(a,4) for a in accs[:3]],'n_train':len(ymt),'n_test':len(yme)}
    fig,axs=plt.subplots(1,4,figsize=(5.3,1.95),dpi=240,sharey=True)
    for k,ax in enumerate(axs):
        if k<3:
            t=bag.estimators_[k]; Z=t.predict(GRIDXY); region(ax,Z); ttl=f'tree {k+1}'; acc=accs[k]
        else:
            Z=bag.predict_proba(GRIDXY)[:,1]; region(ax,Z,hard=False); ttl='average of 100'; acc=bag.score(Xme,yme)
        cls_pts(ax,Xmt,ymt,s=6); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=9,color=NAVY)
        ax.text(0.97,0.04,f'test {acc:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    fig.tight_layout(w_pad=0.3); sv(fig,'f252.png',470)

# ---- 25.3-1 the bootstrap: left-out fraction and the counts
if want('f253'):
    from scipy.stats import binom
    rng=np.random.default_rng(1)
    ns=np.array([2,3,4,6,10,20,50,100,426,1000]); ng=np.unique(np.round(np.logspace(np.log10(2),3,80)).astype(int))
    sim=[np.mean([ (np.bincount(rng.integers(0,n,n),minlength=n)==0).mean() for _ in range(300 if n<=100 else 60)]) for n in ns]
    NUM['boot']={'n':[int(v) for v in ns],'exact':[round(float((1-1/n)**n),4) for n in ns],'sim':[round(float(v),4) for v in sim],'inv_e':round(float(np.exp(-1)),4)}
    n=426; cnt=np.concatenate([np.bincount(rng.integers(0,n,n),minlength=n) for _ in range(400)]); ks=np.arange(0,7)
    emp=np.array([np.mean(cnt==k) for k in ks]); pb=binom.pmf(ks,n,1/n)
    NUM['boot']['counts']={'k':[int(k) for k in ks],'sim':[round(float(v),4) for v in emp],'binom':[round(float(v),4) for v in pb],'poisson':[round(float(np.exp(-1)/math.factorial(int(k))),4) for k in ks],'mean_distinct':round(float(1-np.mean(cnt==0)),4)}
    fig,axs=plt.subplots(1,2,figsize=(5.2,2.3),dpi=240,gridspec_kw={'width_ratios':[1.05,1]})
    ax=axs[0]; ax.plot(ng,(1-1/ng)**ng,color=GREEN,lw=1.7); ax.axhline(np.exp(-1),color=PUR,lw=1.0,ls='--'); ax.plot(ns,sim,'o',color=BLUE,ms=3.8,zorder=5)
    ax.set_xscale('log'); ax.set_ylim(0,0.42); ax.set_xlabel('number of samples $n$'); ax.set_ylabel('fraction left out'); grid(ax)
    ax.text(900,np.exp(-1)+0.012,'$1/e$ = 0.368',color=PUR,fontsize=8.8,ha='right',va='bottom')
    ax.annotate('$n$ = 6: 0.335',(6,(5/6)**6),xytext=(2.3,0.17),fontsize=8.8,color=NAVY,arrowprops=dict(arrowstyle='-',color=GREY,lw=0.7))
    ax.plot([],[],'o',color=BLUE,ms=3.8,label='simulation'); ax.plot([],[],color=GREEN,lw=1.7,label='$(1-1/n)^n$'); ax.legend(loc='lower right',fontsize=8.4,borderaxespad=0.2,handlelength=1.5)
    ax=axs[1]; ax.bar(ks-0.18,emp,0.36,color=BLUE,label='simulation, $n$ = 426'); ax.bar(ks+0.18,np.exp(-1)/np.array([math.factorial(int(k)) for k in ks]),0.36,color=OR,label='$e^{-1}/k!$')
    ax.set_xlabel('times a sample is drawn, $k$'); ax.set_ylabel('share of samples'); ax.set_xticks(ks); ax.set_ylim(0,0.5); ax.legend(loc='upper right',fontsize=8.4,borderaxespad=0.2,handlelength=1.2); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.0); sv(fig,'f253.png',450)

# ---- 25.3-2 variance of an average of B correlated predictors
if want('f254'):
    rng=np.random.default_rng(2); Bs=np.array([1,2,5,10,20,50,100,200]); Bg=np.logspace(0,np.log10(200),100); rhos=[0.0,0.2,0.5,0.8]; sty=[(BLUE,'o','-'),(GREEN,'s','--'),(OR,'^','-.'),(RED,'D',':')]
    NUM['varB']={}
    fig,ax=plt.subplots(figsize=(3.9,2.55),dpi=240)
    for rho,(col,mk,ls) in zip(rhos,sty):
        sim=[]
        for B in Bs:
            Z=rng.standard_normal((40000,1)); E=rng.standard_normal((40000,B)); Xb=np.sqrt(rho)*Z+np.sqrt(1-rho)*E; sim.append(float(Xb.mean(axis=1).var()))
        ax.plot(Bg,rho+(1-rho)/Bg,color=col,lw=1.5,ls=ls); ax.plot(Bs,sim,mk,color=col,ms=4.2,mec='white',mew=0.5,zorder=5)
        ax.text(230,rho+(1-rho)/200+(0.025 if rho==0 else 0),f'$\\rho$ = {rho:g}',color=col,fontsize=8.8,va='center',ha='left')
        NUM['varB'][str(rho)]={'B':[int(b) for b in Bs],'sim':[round(v,4) for v in sim],'formula':[round(float(rho+(1-rho)/b),4) for b in Bs]}
    ax.set_xscale('log'); ax.set_xlim(1,200); ax.set_ylim(0,1.05); ax.set_xticks([1,3,10,30,100]); ax.set_xticklabels(['1','3','10','30','100']); ax.set_xlabel('number of predictors $B$'); ax.set_ylabel('variance of the average (units of $\\sigma^2$)'); grid(ax)
    ax.set_xlim(1,205); fig.subplots_adjust(right=0.86)
    sv(fig,'f254.png',390)

# ---- 25.4-1 decorrelation by max_features (make_friedman1, 60 training sets)
if want('f255'):
    from sklearn.datasets import make_friedman1
    from sklearn.ensemble import RandomForestRegressor as RFR
    Xt,yt=make_friedman1(300,n_features=10,noise=1.0,random_state=999); ms=[1,2,3,5,7,10]; R=60; B=25; out=[]
    for m in ms:
        P=np.zeros((R,B,len(yt)))
        for r in range(R):
            X_,y_=make_friedman1(200,n_features=10,noise=1.0,random_state=r); f=RFR(n_estimators=B,max_features=m,random_state=r).fit(X_,y_); P[r]=np.array([t.predict(Xt) for t in f.estimators_])
        mu=P.mean(axis=(0,1)); D=P-mu; var=(D**2).mean(); S=D.sum(axis=1); cov=((S**2-(D**2).sum(axis=1))/(B*(B-1))).mean()
        fb=P.mean(axis=1); out.append({'m':m,'rho':round(float(cov/var),4),'sigma2':round(float(var),3),'forest_var':round(float(fb.var(axis=0).mean()),3),'formula_var':round(float(cov+(var-cov)/B),3),
            'mse_tree':round(float(((P-yt)**2).mean()),3),'mse_forest':round(float(((fb-yt)**2).mean()),3)})
    NUM['decorr']=out
    fig,axs=plt.subplots(1,2,figsize=(5.0,2.3),dpi=240)
    ax=axs[0]; ax.plot(ms,[o['rho'] for o in out],'-o',color=PUR,ms=4.2,lw=1.5); ax.set_xlabel('max_features $m$'); ax.set_ylabel('correlation $\\rho$ of two trees'); ax.set_ylim(0,0.13); ax.set_xticks(ms); grid(ax)
    ax=axs[1]; ax.plot(ms,[o['mse_tree'] for o in out],'-s',color=GREY,ms=4,lw=1.4,label='one tree'); ax.plot(ms,[o['mse_forest'] for o in out],'-o',color=GREEN,ms=4.2,lw=1.6,label='forest of 25')
    ax.set_xlabel('max_features $m$'); ax.set_ylabel('test mean squared error'); ax.set_xticks(ms); ax.set_ylim(0,34); ax.legend(loc='upper right',fontsize=8.4,borderaxespad=0.2,handlelength=1.6); grid(ax)
    fig.tight_layout(w_pad=1.0); sv(fig,'f255.png',450)

# ---- 25.4-2 best split against random split; extra trees on the moons
if want('f256'):
    inc=np.array([22,40,55,28,62,18,36,26,48,75.]); yl=np.array([0,1,0,0,1,0,1,0,1,1])
    def g_(y): p=y.mean() if len(y) else 0; return 2*p*(1-p)
    def dec(t): L=yl[inc<=t]; R=yl[inc>t]; return g_(yl)-len(L)/10*g_(L)-len(R)/10*g_(R)
    vs=np.sort(inc); mids=(vs[:-1]+vs[1:])/2; dm=np.array([dec(t) for t in mids]); kb=int(np.argmax(dm))
    rr=np.random.default_rng(17); draws=np.sort(rr.uniform(18,75,3)); dd=[dec(t) for t in draws]
    NUM['split']={'mids':[float(m) for m in mids],'dec':[round(float(v),4) for v in dm],'best':float(mids[kb]),'draws':[round(float(v),2) for v in draws],'draw_dec':[round(float(v),4) for v in dd]}
    et=ETC(n_estimators=100,random_state=0).fit(Xmt,ymt); rf=RFC(n_estimators=100,random_state=0).fit(Xmt,ymt)
    NUM['moons_et']={'et_test':round(float(et.score(Xme,yme)),4),'rf_test':round(float(rf.score(Xme,yme)),4)}
    fig,axs=plt.subplots(1,2,figsize=(5.2,2.35),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
    ax=axs[0]; ax.vlines(mids,0,dm,color=GREEN,lw=2.0); ax.plot(mids,dm,'o',color=GREEN,ms=3.8); ax.plot(mids[kb],dm[kb],'*',color=GREEN,ms=12,mec='white',zorder=6)
    for t,d_ in zip(draws,dd): ax.axvline(t,color=PUR,lw=1.0,ls='--'); ax.plot(t,d_,'v',color=PUR,ms=6,mec='white',zorder=6)
    ax.set_xlabel('threshold on income'); ax.set_ylabel('impurity decrease $\\Delta$'); ax.set_ylim(0,0.52); ax.set_xlim(14,80); grid(ax)
    ax.plot([],[],'-o',color=GREEN,ms=3.8,lw=2.0,label='the 9 midpoints; best starred'); ax.plot([],[],'--v',color=PUR,ms=5,lw=1.0,label='3 random draws'); ax.legend(loc='upper right',fontsize=8.2,borderaxespad=0.1,handlelength=1.8)
    ax=axs[1]; Z=et.predict_proba(GRIDXY)[:,1]; region(ax,Z,hard=False); cls_pts(ax,Xmt,ymt,s=6); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('extra trees, 100',fontsize=9,color=NAVY)
    ax.text(0.97,0.04,f'test {et.score(Xme,yme):.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    fig.tight_layout(w_pad=1.0); sv(fig,'f256.png',450)

# ---- 25.5-1 OOB error and test error against the number of trees (5 seeds, 400 trees)
if want('f257'):
    cd=load_breast_cancer(); Xt_,Xe_,yt_,ye_=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
    Bmax=400; seeds=range(5); te=np.zeros((5,Bmax)); oo=np.full((5,Bmax),np.nan)
    for si,sd in enumerate(seeds):
        f=RFC(n_estimators=Bmax,random_state=sd,n_jobs=1).fit(Xt_,yt_); Pte=np.cumsum([t.predict_proba(Xe_) for t in f.estimators_],axis=0)
        te[si]=[ (Pte[b].argmax(1)!=ye_).mean() for b in range(Bmax)]
        Po=np.zeros((len(yt_),2)); mo=np.zeros(len(yt_))
        for b,(t,idx) in enumerate(zip(f.estimators_,f.estimators_samples_)):
            out=np.bincount(idx,minlength=len(yt_))==0; Po[out]+=t.predict_proba(Xt_[out]); mo[out]+=1
            if b+1>=20 and (mo>0).all(): oo[si,b]=(Po.argmax(1)!=yt_).mean()
    Bx=np.arange(1,Bmax+1); NUM['oobcurve']={'B':[10,25,50,100,200,400],'oob_mean':[round(float(np.nanmean(oo[:,b-1])),4) if not np.isnan(oo[:,b-1]).all() else None for b in [10,25,50,100,200,400]],'test_mean':[round(float(te[:,b-1].mean()),4) for b in [10,25,50,100,200,400]],'test_B1':round(float(te[:,0].mean()),4),
        'oob_range_100plus':[round(float(np.nanmin(oo[:,99:])),4),round(float(np.nanmax(oo[:,99:])),4)],'test_range_100plus':[round(float(te[:,99:].min()),4),round(float(te[:,99:].max()),4)],'oob_mean_100plus':round(float(np.nanmean(oo[:,99:])),4),'test_mean_100plus':round(float(te[:,99:].mean()),4)}
    fig,ax=plt.subplots(figsize=(4.4,2.5),dpi=240)
    ax.fill_between(Bx,te.min(0),te.max(0),color=OR,alpha=0.18,lw=0); ax.plot(Bx,te.mean(0),color=OR,lw=1.5,ls='-',label='test error (143 patients)')
    ok=~np.isnan(oo).any(0); NUM['oobcurve']['first_B_all_defined']=int(Bx[ok][0]); ax.fill_between(Bx[ok],oo[:,ok].min(0),oo[:,ok].max(0),color=BLUE,alpha=0.18,lw=0); ax.plot(Bx[ok],oo[:,ok].mean(0),color=BLUE,lw=1.5,ls='--',label='out-of-bag error (426 patients)')
    ax.set_xscale('log'); ax.set_xlim(1,400); ax.set_ylim(0.02,0.14); ax.set_xticks([1,3,10,30,100,300]); ax.set_xticklabels(['1','3','10','30','100','300']); ax.set_xlabel('number of trees $B$'); ax.set_ylabel('error rate'); grid(ax); ax.legend(loc='upper right',fontsize=8.4,borderaxespad=0.2,handlelength=2.0)
    sv(fig,'f257.png',420)

# ---- 25.6-1 impurity importance against permutation importance (as code/importance_bias.py)
if want('f258'):
    from sklearn.inspection import permutation_importance
    rng=np.random.default_rng(0); n=1000; s_=rng.integers(0,3,n); u_=rng.uniform(size=n); v_=rng.integers(0,2,n); y_=rng.random(n)<0.2+0.3*s_
    X_=np.c_[s_,u_,v_]; Xa,Xb,ya,yb=X_[:500],X_[500:],y_[:500],y_[500:]
    f=RFC(n_estimators=300,random_state=0).fit(Xa,ya); pm=permutation_importance(f,Xb,yb,n_repeats=20,random_state=0)
    NUM['impbias']={'mdi':[round(float(v),4) for v in f.feature_importances_],'perm_mean':[round(float(v),4) for v in pm.importances_mean],'perm_std':[round(float(v),4) for v in pm.importances_std],'train':float(f.score(Xa,ya)),'test':round(float(f.score(Xb,yb)),4)}
    fig,axs=plt.subplots(1,2,figsize=(4.4,2.1),dpi=240)
    names=['$s$\n3 values','$u$\n1000 values','$v$\n2 values']; cols=[GREEN,GREY,GREY]
    ax=axs[0]; ax.bar(range(3),f.feature_importances_,color=cols,width=0.62)
    for k,v in enumerate(f.feature_importances_): ax.text(k,v+0.015,f'{v:.2f}',ha='center',fontsize=8.6,color=NAVY)
    ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=8.4); ax.set_ylim(0,0.9); ax.set_title('impurity (training)',fontsize=9,color=NAVY); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
    ax=axs[1]; ax.bar(range(3),pm.importances_mean,yerr=pm.importances_std,color=cols,width=0.62,error_kw=dict(lw=0.9,capsize=2.5,ecolor=NAVY))
    ax.axhline(0,color=GREY,lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels(names,fontsize=8.4); ax.set_title('permutation (test)',fontsize=9,color=NAVY); ax.set_ylim(-0.05,0.16); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2); sv(fig,'f258.png',380)

# ---- 25.6-2 random trees embedding on two circles (as code/embedding.py)
if want('f259'):
    from sklearn.datasets import make_circles
    from sklearn.ensemble import RandomTreesEmbedding
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    Xc,yc=make_circles(n_samples=400,noise=0.1,factor=0.4,random_state=0); Xa,Xb,ya,yb=train_test_split(Xc,yc,test_size=0.5,random_state=0,stratify=yc)
    raw=LogisticRegression().fit(Xa,ya); emb=RandomTreesEmbedding(n_estimators=20,max_depth=3,random_state=0); pipe=make_pipeline(emb,LogisticRegression(max_iter=1000)).fit(Xa,ya)
    NUM['embed']={'raw_test':round(float(raw.score(Xb,yb)),4),'emb_test':round(float(pipe.score(Xb,yb)),4)}
    g1,g2=np.meshgrid(np.linspace(-1.6,1.6,260),np.linspace(-1.6,1.6,260)); G=np.c_[g1.ravel(),g2.ravel()]
    fig,axs=plt.subplots(1,2,figsize=(4.3,2.15),dpi=240,sharey=True)
    for ax,m,ttl in zip(axs,[raw,pipe],['logistic regression on $(x_1,x_2)$','on the leaf indicators']):
        Z=m.predict(G); ax.contourf(g1,g2,Z.reshape(g1.shape),levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(g1,g2,Z.reshape(g1.shape),levels=[0.5],colors=[GREEN],linewidths=1.4)
        cls_pts(ax,Xa,ya,s=8); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=8.6,color=NAVY); ax.set_aspect('equal')
        ax.text(0.97,0.04,f'test {m.score(Xb,yb):.3f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    fig.tight_layout(w_pad=0.4); sv(fig,'f259.png',400)

# ---- 25.7-1 by hand: three stumps on three bootstrap samples
if want('f257h'):
    fig,axs=plt.subplots(1,3,figsize=(5.3,2.05),dpi=240,sharey=True)
    for k,(ax,d,st,pp) in enumerate(zip(axs,DRAWS,stumps,PS)):
        w=np.bincount(d,minlength=6); ax.plot(XG[:,0],pp,color=GREEN,lw=1.8,zorder=2)
        for i in range(6):
            col=[BLUE,OR][REP[i]]; mk='os'[REP[i]]
            if w[i]>0: ax.scatter(INC[i],REP[i],s=34,color=col,marker=mk,edgecolor='white',lw=0.7,zorder=5,clip_on=False)
            else: ax.scatter(INC[i],REP[i],s=34,facecolor='white',marker=mk,edgecolor=col,lw=1.1,zorder=5,clip_on=False)
            if w[i]>1: ax.text(INC[i],REP[i]+(0.16 if REP[i]==0 else -0.16),f'$\\times${w[i]}',fontsize=8.4,color=NAVY,ha='center',va='center')
        ax.set_xlim(12,68); ax.set_ylim(-0.3,1.3); ax.set_xticks(INC); ax.set_xticklabels([str(i+1) for i in range(6)]); ax.set_xlabel('applicant number',fontsize=8.8); ax.set_title(f'sample {k+1}: split at {st.tree_.threshold[0]:.1f}',fontsize=8.8,color=NAVY); ax.set_yticks([0,0.5,1]); grid(ax)
    axs[0].set_ylabel('$P$(repaid)'); fig.tight_layout(w_pad=0.4); sv(fig,'f257h.png',470)

# ---- 25.8-1 flow chart of fit and predict
if want('f25flow'):
    fig,ax=new(3.8,4.2)
    def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
    A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
    def Ln(*pts): ax.plot([p[0] for p in pts],[p[1] for p in pts],color=NAVY,lw=0.9)
    def dia(xc,yc,w,h,t,fs=8.2): ax.add_patch(Polygon([(xc,yc+h/2),(xc+w/2,yc),(xc,yc-h/2),(xc-w/2,yc)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.1)
    def yn(x,y,t,ha='left'): ax.text(x,y,t,fontsize=8,color=GREY,style='italic',ha=ha,va='center')
    X0=2.6
    fb(X0,7.75,4.8,0.85,'Input:  $\\mathbf{X}$,  $y$,  $B$,  $m$ = max_features;\none seed per tree from random_state',fc=LGREY,ec=GREY)
    dia(X0,6.35,3.0,1.05,'$b \\leq B$ ?')
    fb(X0,4.8,4.8,1.3,'Draw $n$ indices with replacement;\ncount draws: $c_i$ = weight of sample $i$\n(out of bag if $c_i = 0$)')
    fb(X0,3.1,4.8,1.3,'Grow tree $b$ with weights $c_i$, searching\n$m$ random features at each split\n(Chapter 24); store it;  $b \\leftarrow b + 1$')
    fb(X0,0.95,4.8,1.3,'Out of bag: average, for each sample, the\ntrees with $c_i = 0$ (oob_score_);\npredict: average the $B$ trees, take argmax',fc='#E3F4EA',ec=GREEN)
    A((X0,7.325),(X0,6.875)); A((X0,5.825),(X0,5.45)); A((X0,4.15),(X0,3.75))
    Ln((X0-2.4,3.1),(0.0,3.1),(0.0,6.35),(X0-1.5,6.35)); A((X0-1.5,6.35),(X0-1.45,6.35))
    Ln((X0+1.5,6.35),(5.35,6.35),(5.35,0.95),(X0+2.4,0.95)); A((X0+2.4,0.95),(X0+2.39,0.95))
    yn(X0+0.12,5.65,'yes'); yn(X0+1.7,6.55,'no')
    ax.set_xlim(-0.2,5.5); ax.set_ylim(0.1,8.3); sv(fig,'f25flow.png',360)

# ---- 25.11-1 gallery: min_samples_leaf on the moons
if want('f2511a'):
    fig,axs=plt.subplots(1,4,figsize=(5.2,1.95),dpi=240,sharey=True); NUM['gallery']={}
    for ax,msl in zip(axs,[1,5,15,40]):
        m=RFC(n_estimators=100,min_samples_leaf=msl,random_state=0).fit(Xmt,ymt); Z=m.predict_proba(GRIDXY)[:,1]; region(ax,Z,hard=False); cls_pts(ax,Xmt,ymt,s=6)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'leaf size {msl}',fontsize=9,color=NAVY); acc=m.score(Xme,yme)
        ax.text(0.97,0.04,f'test {acc:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
        NUM['gallery'][str(msl)]={'train':round(float(m.score(Xmt,ymt)),4),'test':round(float(acc),4)}
    fig.tight_layout(w_pad=0.3); sv(fig,'f2511a.png',470)

# ---- 25.11-2 validation curves: max_features and max_samples on the breast-cancer training set
if want('f2511b'):
    cd=load_breast_cancer(); Xt_,Xe_,yt_,ye_=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
    V={}
    for key,grid_ in (('max_features',[1,2,3,5,8,12,20,30]),('max_samples',[0.1,0.2,0.4,0.6,0.8,1.0])):
        cvm=[];cvs=[];oo=[]
        for g in grid_:
            kw={key:g}; sc=cross_val_score(RFC(n_estimators=100,random_state=0,**kw),Xt_,yt_,cv=5); cvm.append(float(sc.mean())); cvs.append(float(sc.std()))
            oo.append(float(RFC(n_estimators=100,oob_score=True,random_state=0,**kw).fit(Xt_,yt_).oob_score_))
        V[key]={'grid':grid_,'cv':[round(v,4) for v in cvm],'sd':[round(v,4) for v in cvs],'oob':[round(v,4) for v in oo]}
    NUM['valcurve']=V
    fig,axs=plt.subplots(1,2,figsize=(5.2,2.3),dpi=240)
    for ax,key in zip(axs,['max_features','max_samples']):
        g=np.array(V[key]['grid'],float); cv=np.array(V[key]['cv']); sd=np.array(V[key]['sd']); oo=np.array(V[key]['oob']); k=int(np.argmax(cv))
        ax.plot(g,cv,'-s',color=GREY,ms=3.4,lw=1.4,label='5-fold validation'); ax.fill_between(g,cv-sd,cv+sd,color=GREY,alpha=0.18,lw=0); ax.plot(g,oo,'--o',color=BLUE,ms=3.4,lw=1.3,label='out of bag')
        ax.axvline(g[k],color=PUR,lw=1.0,ls='--'); ax.set_xlabel(key); ax.set_ylim(0.90,1.0); grid(ax)
        if key=='max_features': ax.set_xscale('log'); ax.set_xticks([1,2,5,10,30]); ax.set_xticklabels(['1','2','5','10','30']); ax.axvline(5,color=GRID,lw=0.9,ls=':'); ax.text(5.4,0.997,'default $\\sqrt{p}$',fontsize=8.2,color=GREY,va='top')
        else: ax.set_xticks([0.1,0.4,0.7,1.0])
    axs[0].set_ylabel('accuracy'); axs[0].legend(loc='lower left',fontsize=8.2,borderaxespad=0.2,handlelength=1.8)
    fig.tight_layout(w_pad=1.0); sv(fig,'f2511b.png',450)

# ---- 25.12-1 case study summary, read from the captured outputs of code/case_cancer.py and code/case_diabetes.py
if want('f2512'):
    import re
    def parse(fn,nm):
        rows=[]
        for ln in open(fn).read().splitlines()[1:]:
            m=re.match(r'^(.{20,30}?)\s{2,}([0-9.]+)\s+([0-9.]+)\s+(?:([0-9.]+)\s+)?([0-9.]+|-)$',ln) if nm=='d' else re.match(r'^(.{20}?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+|-)$',ln)
            if m: rows.append(m.groups())
        return rows
    cc=[l for l in open('verify/case_cancer.out').read().splitlines()[1:6]]; Rc=[]
    for ln in cc:
        nm=ln[:20].strip(); a=ln[20:].split(); Rc.append((nm,float(a[0]),float(a[1]),None if a[2]=='-' else float(a[2])))
    dd=[l for l in open('verify/case_diabetes.out').read().splitlines()[1:7]]; Rd=[]
    for ln in dd:
        nm=ln[:28].strip(); a=ln[28:].split(); Rd.append((nm,float(a[0]),float(a[1]),None if a[3]=='-' else float(a[3])))
    NUM['case']={'cancer':Rc,'diabetes':Rd}
    fig,axs=plt.subplots(1,2,figsize=(5.3,2.5),dpi=240,gridspec_kw={'width_ratios':[1,1.08]})
    for ax,R,ttl,lo,hi in ((axs[0],Rc,'breast cancer: accuracy',0.89,0.985),(axs[1],Rd,'diabetes: $R^2$',0.05,0.56)):
        yy=np.arange(len(R))[::-1]
        for yv,(nm,cv,te,oo) in zip(yy,R):
            ax.plot([min(cv,te),max(cv,te)],[yv,yv],color=GRID,lw=1.2,zorder=1)
            ax.scatter(cv,yv,marker='s',s=26,color=GREY,zorder=5,label='5-fold' if yv==yy[0] else None); ax.scatter(te,yv,marker='^',s=32,color=OR,zorder=5,label='test' if yv==yy[0] else None)
            if oo is not None: ax.scatter(oo,yv,marker='o',s=24,color=BLUE,zorder=5,label='out of bag' if oo is not None and yv==yy[[k for k,r_ in enumerate(R) if r_[3] is not None][0]] else None)
        ax.set_yticks(yy); ax.set_yticklabels([r_[0].replace('logistic regression','logistic regr.').replace('one tree, leaf 20','tree, leaf 20').replace('forest, m = p/3, leaf 5','forest, p/3, leaf 5').replace('extra trees, m = p/3, leaf 5','extra, p/3, leaf 5').replace('forest, m = p/3','forest, p/3').replace('forest, m = p','forest, m = p') for r_ in R],fontsize=8.4)
        ax.set_xlim(lo,hi); ax.set_ylim(-0.6,len(R)-0.4); ax.set_title(ttl,fontsize=9,color=NAVY); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
    axs[0].legend(loc='lower left',fontsize=8.2,borderaxespad=0.2,handletextpad=0.2,labelspacing=0.2,borderpad=0.2); 
    fig.tight_layout(w_pad=0.6); sv(fig,'f2512.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)

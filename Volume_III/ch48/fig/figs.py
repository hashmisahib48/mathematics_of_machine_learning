# All figures of Chapter 48. Run: python3 fig/figs.py  (PNGs into ch48/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score
from sensors import make_sensors
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
X,Y=make_sensors(); sc=StandardScaler().fit(X); Z=sc.transform(X)

# ======== 48.0-1 opener: outlier detection and novelty detection
fig,axs=plt.subplots(1,2,figsize=(5.0,1.95),dpi=240,sharex=True,sharey=True)
ax=axs[0]; f=IsolationForest(contamination=0.05,random_state=0).fit(Z); lab=f.predict(Z)==-1
ax.scatter(X[~lab,0],X[~lab,1],s=5,color=BLUE,lw=0,alpha=0.8); ax.scatter(X[lab,0],X[lab,1],s=22,marker='x',color=RED,lw=1.1)
ax.set_title('outlier detection (unclean data)',fontsize=8.5,color=NAVY); ax.set_xlabel('temperature (°C)'); ax.set_ylabel('vibration (mm/s)'); grid(ax)
NUM['open_left']={'flagged':int(lab.sum()),'n':len(X)}
ax=axs[1]; Xc=X[Y==0]; m=OneClassSVM(nu=0.05).fit(sc.transform(Xc))
gx,gy=np.meshgrid(np.linspace(38,92,260),np.linspace(-0.2,5.4,260)); G=m.decision_function(sc.transform(np.c_[gx.ravel(),gy.ravel()])).reshape(gx.shape)
ax.contourf(gx,gy,G,levels=[0,10],colors=['#E3F4EA'],zorder=0); ax.contour(gx,gy,G,levels=[0],colors=[GREEN],linewidths=1.6,zorder=1)
ax.scatter(Xc[:,0],Xc[:,1],s=4,color=BLUE,lw=0,alpha=0.35,zorder=2)
Nw=np.array([[54,1.2],[58,0.8],[73,2.9],[78,2.2],[62,1.3],[60,2.6],[66,4.4],[85,1.0],[48,3.2]]); v=m.predict(sc.transform(Nw))
ax.scatter(*Nw[v==1].T,s=26,facecolor='white',edgecolor=BLUE,lw=1.2,zorder=3); ax.scatter(*Nw[v==-1].T,s=26,marker='x',color=RED,lw=1.3,zorder=3)
ax.set_title('novelty detection (new readings)',fontsize=8.5,color=NAVY); ax.set_xlabel('temperature (°C)'); ax.set_xlim(38,92); ax.set_ylim(-0.2,5.4); grid(ax)
NUM['open_right']={'new_flagged':int((v==-1).sum()),'new':len(Nw)}
fig.tight_layout(w_pad=0.6); sv(fig,'f480.png',470)

# ======== 48.2-1 intuition: isolation by random cuts
def isolate(P,t,rng,bbx):
    # random axis-parallel cuts until point t is alone; returns the cuts (axis, value, cell box) and the final box
    idx=np.arange(len(P)); cuts=[]; lo=np.array(bbx[:2],float); hi=np.array(bbx[2:],float)
    while len(idx)>1:
        a=rng.randint(2); mn,mx=P[idx,a].min(),P[idx,a].max()
        if mn==mx: break
        c=rng.uniform(mn,mx); cuts.append((a,c,lo.copy(),hi.copy()))
        if P[t,a]<c: hi[a]=c; idx=idx[P[idx,a]<c]
        else: lo[a]=c; idx=idx[P[idx,a]>=c]
    return cuts,lo,hi
rs=np.random.RandomState(4); sub=rs.choice(np.where(Y==0)[0],60,replace=False); S=np.vstack([X[sub],[[66.0,4.3]]]); S=(S-S.mean(0))/S.std(0)
t_out=len(S)-1; dist=np.hypot(S[:,0][:,None]-S[:,0][None],S[:,1][:,None]-S[:,1][None]); t_in=int(np.argsort(dist.sum(1))[3])
bbx=(-2.6,-1.9,2.6,3.2)
SEED_O=next(j for j in range(100) if len(isolate(S,t_out,np.random.RandomState(j),bbx)[0])==3)
SEED_I=next(j for j in range(100) if len(isolate(S,t_in,np.random.RandomState(j),bbx)[0])==9)
fig,axs=plt.subplots(1,2,figsize=(5.0,2.1),dpi=240,sharex=True,sharey=True)
for ax,t,ttl,seed in ((axs[0],t_out,'an outlier',SEED_O),(axs[1],t_in,'a point in the crowd',SEED_I)):
    ax.scatter(S[:,0],S[:,1],s=6,color=BLUE,lw=0,alpha=0.7)
    cuts,lo,hi=isolate(S,t,np.random.RandomState(seed),bbx); ax.add_patch(Rectangle(lo,*(hi-lo),fc='#E3F4EA',ec='none',zorder=0))
    for a,c,l,h in cuts:
        if a==0: ax.plot([c,c],[l[1],h[1]],color=PUR,lw=0.8,zorder=2)
        else: ax.plot([l[0],h[0]],[c,c],color=PUR,lw=0.8,zorder=2)
    ax.scatter(*S[t],s=34,marker='o',facecolor=RED,edgecolor='white',lw=0.6,zorder=4)
    ax.set_xlim(bbx[0],bbx[2]); ax.set_ylim(bbx[1],bbx[3]); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{ttl}: isolated by {len(cuts)} cuts',fontsize=8.5,color=NAVY)
    for sp_ in ax.spines.values(): sp_.set_visible(True)
hs={}
for nm,t in (('outlier',t_out),('crowd',t_in)): hs[nm]=float(np.mean([len(isolate(S,t,np.random.RandomState(1000+j),bbx)[0]) for j in range(2000)]))
NUM['isolate_mean_cuts']=hs
fig.tight_layout(w_pad=0.6); sv(fig,'f481.png',410)

# ======== 48.2-2 the toy readings and three verdicts
xt=np.array([0,1,2,3,7,13,17,21.])[:,None]; nt=len(xt)
dd,_=NearestNeighbors(n_neighbors=3).fit(xt).kneighbors(xt); knn=dd[:,1:].mean(1)
lofv=-LocalOutlierFactor(n_neighbors=2).fit(xt).negative_outlier_factor_
isov=-IsolationForest(n_estimators=1000,max_samples=8,random_state=0).fit(xt).score_samples(xt)
NUM['toy']={'knn':knn.round(3).tolist(),'lof':lofv.round(4).tolist(),'iso':isov.round(3).tolist()}
fig=plt.figure(figsize=(5.0,2.3),dpi=240); gs=fig.add_gridspec(2,3,height_ratios=[0.4,1.3],hspace=0.55,wspace=0.35)
ax=fig.add_subplot(gs[0,:]); ax.axhline(0,color=GREY,lw=0.8); ax.scatter(xt[:,0],np.zeros(nt),s=26,color=BLUE,zorder=3)
for v in xt[:,0]: ax.text(v,0.55,f'{v:.0f}',ha='center',va='bottom',fontsize=8,color=NAVY)
ax.set_xlim(-1,22); ax.set_ylim(-0.6,1.5); ax.axis('off'); ax.set_title('eight readings of vibration (mm/s)',fontsize=8.5,color=NAVY,pad=0)
for j,(vals,ttl,lab) in enumerate(((knn,'mean distance to\ntwo neighbours',None),(lofv,'local outlier factor\n($k=2$)',1.0),(isov,'isolation score\n($\\psi=8$)',0.5))):
    ax=fig.add_subplot(gs[1,j]); top=int(np.argmax(vals)); cols=[RED if q==top else BLUE for q in range(nt)]
    ax.bar(range(nt),vals,color=cols,width=0.7); ax.set_xticks(range(nt)); ax.set_xticklabels([f'{v:.0f}' for v in xt[:,0]],fontsize=7.5); ax.tick_params(axis='x',length=2,pad=1.5)
    if lab is not None: ax.axhline(lab,color=GREEN,lw=1.0,ls='--')
    ax.set_title(ttl,fontsize=8.5,color=NAVY); ax.set_xlabel('reading',labelpad=1); grid(ax)
sv(fig,'f482.png',470)

# ======== 48.4-1 isolation score
def cpath(m): return 0 if m<=1 else 1 if m==2 else 2*(np.log(m-1)+np.euler_gamma)-2*(m-1)/m
fig,axs=plt.subplots(1,2,figsize=(4.6,2.2),dpi=240)
ax=axs[0]; hh=np.linspace(0,12,300)
for psi,col in ((256,BLUE),(8,OR)):
    c_=cpath(psi); ax.plot(hh,2**(-hh/c_),color=col,lw=1.5,label=f'$\\psi={psi}$'); ax.plot(c_,0.5,'o',color=col,ms=4.5)
ax.axhline(0.5,color=GREEN,lw=0.9,ls='--'); ax.set_xlabel('mean path length $\\mathrm{E}\\,h(\\mathbf{x})$'); ax.set_ylabel('score $s$'); ax.legend(loc='upper right',handlelength=1.2); ax.set_ylim(0,1.02); grid(ax)
ax.text(11.8,0.53,'0.5',color=GREEN,fontsize=9,ha='right')
ax=axs[1]; f=IsolationForest(random_state=0).fit(Z); s_=-f.score_samples(Z); bins=np.linspace(0.3,0.8,31)
ax.hist(s_[Y==0],bins=bins,color=BLUE,alpha=0.75,label='normal'); ax.hist(s_[Y==1],bins=bins,color=RED,alpha=0.85,label='faulty')
ax.axvline(0.5,color=GREEN,lw=1.2,ls='--'); ax.set_xlabel('score $s$ of the 315 readings'); ax.set_ylabel('readings'); ax.legend(loc='upper right',handlelength=1.2); grid(ax)
NUM['iforest_hist']={'normal_above_half':int((s_[Y==0]>0.5).sum()),'fault_above_half':int((s_[Y==1]>0.5).sum()),'n_normal':int((Y==0).sum())}
fig.tight_layout(w_pad=0.8); sv(fig,'f483.png',410)

# ======== 48.6-1 flow chart
fig,ax=plt.subplots(figsize=(5.0,2.45),dpi=240); ax.set_xlim(0,10); ax.set_ylim(-0.6,7.4); ax.axis('off')
def row(y,name,items):
    ax.text(0.0,y+0.95,name,fontsize=8.5,color=RED,weight='bold',va='center')
    x=0.0; gap=0.3
    for k_,(w_,txt,kind) in enumerate(items):
        fc,ec={'d':(LB,BLUE),'s':(LGREY,GREY),'o':('#E3F4EA',GREEN),'m':(PINK,RED)}[kind]
        box(ax,x,y-0.5,w_,1.0,txt,fc=fc,ec=ec,fs=7.8); x+=w_
        if k_<len(items)-1: arr(ax,(x,y),(x+gap,y),c=NAVY,lw=1,ms=7); x+=gap
row(6.2,'IsolationForest',[(0.9,'data\n$\\mathbf{X}$','d'),(2.2,'$t$ random\nsub-samples $\\psi$','s'),(2.3,'random cuts\nuntil isolated','s'),(2.0,'$s=2^{-\\mathrm{E}h/c(\\psi)}$\nEq. 48.4-3','o'),(0.9,'cut\n$s$','m')])
row(4.2,'LocalOutlierFactor',[(0.9,'data\n$\\mathbf{X}$','d'),(1.9,'$k$ nearest\nneighbours','s'),(1.8,'reach-dist,\nlrd','s'),(2.2,'LOF = mean lrd(b)\n/ lrd(a)','o'),(1.2,'cut\nLOF','m')])
row(2.2,'OneClassSVM',[(0.9,'data\n$\\mathbf{X}$','d'),(1.9,'kernel\nmatrix $\\mathbf{K}$','s'),(2.2,'QP for $\\boldsymbol{\\alpha}$\nwith $\\nu$','s'),(1.9,'$f=\\sum\\alpha_i k-\\rho$','o'),(1.0,'sign\n$f$','m')])
row(0.2,'EllipticEnvelope',[(0.9,'data\n$\\mathbf{X}$','d'),(1.9,'MinCovDet\n(Chapter 46)','s'),(1.8,'$\\hat{\\boldsymbol{\\mu}}$, $\\hat{\\boldsymbol{\\Sigma}}$','s'),(2.2,'Mahalanobis\n$d^2(\\mathbf{x})$','o'),(1.3,'percentile\ncut','m')])
sv(fig,'f484.png',410)


# ======== 48.8-1 exact one-class SVM, Nystroem + SGD, and time
from sklearn.pipeline import make_pipeline
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
Xb,Yb=make_sensors(n=3000,n_fault=150,seed=1); scb=StandardScaler().fit(Xb); Zb=scb.transform(Xb); gb=1/(2*Zb.var())
gx,gy=np.meshgrid(np.linspace(38,92,200),np.linspace(-0.2,5.4,200)); GZ=scb.transform(np.c_[gx.ravel(),gy.ravel()])
fig,axs=plt.subplots(1,3,figsize=(5.0,2.1),dpi=240,gridspec_kw={'width_ratios':[1,1,1.15]})
sub_=np.random.RandomState(0).choice(len(Xb),500,replace=False); agree={}
for ax,(ttl,m) in zip(axs[:2],(('exact OneClassSVM',OneClassSVM(nu=0.05,gamma=gb)),('Nystroem(100) + SGD',make_pipeline(Nystroem(gamma=gb,n_components=100,random_state=0),SGDOneClassSVM(nu=0.05,tol=None,max_iter=30,random_state=0))))):
    m.fit(Zb); G=m.decision_function(GZ).reshape(gx.shape); agree[ttl]=(G>=0)
    ax.scatter(Xb[sub_,0],Xb[sub_,1],s=2.5,color=BLUE,lw=0,alpha=0.5); ax.contour(gx,gy,G,levels=[0],colors=[GREEN],linewidths=1.5)
    ax.set_title(ttl,fontsize=8.5,color=NAVY); ax.set_xlabel('temperature (°C)',fontsize=8.5); ax.set_xticks([40,60,80]); ax.set_ylim(-0.2,5.4); grid(ax)
axs[1].set_yticklabels([]); axs[0].set_ylabel('vibration (mm/s)')
NUM['sgd_region_agree']=float((agree['exact OneClassSVM']==agree['Nystroem(100) + SGD']).mean())
T=json.load(open('verify/timing.json')); ns=[int(k) for k in T]; ax=axs[2]
for j,(nm,col,mk) in enumerate((('exact',RED,'^'),('approx.',BLUE,'o'))):
    xs=[n for n in ns if T[str(n)][j] is not None]; ax.plot(xs,[T[str(n)][j] for n in xs],'-'+mk,color=col,ms=3.5,lw=1.2,label=nm)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('training samples $n$',fontsize=8.5); ax.set_ylabel('fit time (s)'); ax.legend(loc='upper left',handlelength=1.4,labelspacing=0.2); grid(ax)
fig.tight_layout(w_pad=0.5); sv(fig,'f485.png',470)

# ======== 48.9-1 gallery: four detectors on four data sets
from sklearn.datasets import make_moons, make_circles, make_blobs
def with_outliers(Xin,n_out,seed,d0):
    rng=np.random.RandomState(seed); lo,hi=Xin.min(0)-0.35*np.ptp(Xin,0),Xin.max(0)+0.35*np.ptp(Xin,0); O=[]
    while len(O)<n_out:
        q=rng.uniform(lo,hi)
        if np.sqrt(((Xin-q)**2).sum(1)).min()>d0: O.append(q)
    return np.vstack([Xin,O]),np.r_[np.zeros(len(Xin)),np.ones(n_out)]
rng=np.random.RandomState(0)
D1=rng.multivariate_normal([0,0],[[1,0.85],[0.85,1]],250)
D2=np.vstack([rng.normal([-2,0],0.35,(150,2)),rng.normal([2.5,0.0],1.0,(100,2))])
D3=make_moons(250,noise=0.07,random_state=0)[0]; D4=make_circles(250,noise=0.04,factor=0.4,random_state=0)[0]
DS=[('ellipse',with_outliers(D1,13,1,1.1)),('two blobs',with_outliers(D2,13,2,1.0)),('moons',with_outliers(D3,13,3,0.35)),('circles',with_outliers(D4,13,4,0.3))]
METH=[('IsolationForest',lambda:IsolationForest(contamination=0.05,random_state=0)),('LOF',lambda:LocalOutlierFactor(novelty=True,contamination=0.05)),('OneClassSVM',lambda:OneClassSVM(nu=0.05)),('EllipticEnvelope',lambda:EllipticEnvelope(contamination=0.05,random_state=0))]
fig,axs=plt.subplots(4,4,figsize=(5.0,3.8),dpi=240); GA={}
for i,(dn,(P,yo)) in enumerate(DS):
    P=StandardScaler().fit_transform(P); lim=(P.min(0)-0.5,P.max(0)+0.5); gx,gy=np.meshgrid(np.linspace(lim[0][0],lim[1][0],160),np.linspace(lim[0][1],lim[1][1],160))
    for j,(mn,mk) in enumerate(METH):
        ax=axs[i,j]; d=mk().fit(P); G=d.decision_function(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
        sc_=-d.negative_outlier_factor_ if mn=='LOF' else -d.score_samples(P); auc=roc_auc_score(yo,sc_); GA[f'{dn}/{mn}']=round(float(auc),3)
        ax.contourf(gx,gy,G,levels=np.linspace(min(G.min(),-1e-9),0,6),cmap='Blues_r',alpha=0.55,zorder=0); ax.contourf(gx,gy,G,levels=[0,max(G.max(),1e-9)],colors=['#E3F4EA'],zorder=0); ax.contour(gx,gy,G,levels=[0],colors=[GREEN],linewidths=1.2,zorder=1)
        ax.scatter(P[yo==0,0],P[yo==0,1],s=2.5,color=BLUE,lw=0,zorder=2); ax.scatter(P[yo==1,0],P[yo==1,1],s=9,marker='x',color=RED,lw=0.8,zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(lim[0][0],lim[1][0]); ax.set_ylim(lim[0][1],lim[1][1])
        ax.text(0.97,0.04,f'{auc:.2f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color=NAVY,bbox=dict(fc='white',ec='none',pad=0.8,alpha=0.85),zorder=5)
        for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_linewidth(0.5)
        if i==0: ax.set_title(mn,fontsize=8.5,color=NAVY,pad=3)
        if j==0: ax.set_ylabel(dn,fontsize=8.5)
NUM['gallery_auc']=GA
fig.tight_layout(pad=0.3,w_pad=0.3,h_pad=0.3); sv(fig,'f486.png',470)

# ======== 48.10-1 knobs: nu and gamma of the one-class SVM, then validation curves
fig=plt.figure(figsize=(5.0,3.1),dpi=240); gs=fig.add_gridspec(2,3,hspace=0.55,wspace=0.38,height_ratios=[1.05,1])
gx,gy=np.meshgrid(np.linspace(38,92,200),np.linspace(-0.2,5.4,200)); GZ=sc.transform(np.c_[gx.ravel(),gy.ravel()]); HN={}
for j,nu in enumerate((0.02,0.1,0.4)):
    ax=fig.add_subplot(gs[0,j]); m=OneClassSVM(nu=nu,tol=1e-6).fit(Z); G=m.decision_function(GZ).reshape(gx.shape)
    ax.scatter(X[:,0],X[:,1],s=3.5,color=BLUE,lw=0,alpha=0.7); ax.scatter(X[m.support_,0],X[m.support_,1],s=14,facecolor='none',edgecolor=OR,lw=0.6)
    ax.contour(gx,gy,G,levels=[0],colors=[GREEN],linewidths=1.4); out=(m.decision_function(Z)<-1e-6).mean(); sv_=len(m.support_)/len(Z); HN[nu]=(round(float(out),3),round(float(sv_),3))
    ax.set_title(f'$\\nu={nu}$',fontsize=9,color=NAVY,pad=2); ax.set_xticks([40,60,80]); ax.set_yticks([0,2,4]); ax.set_xlabel(f'outside {out:.2f}, SV {sv_:.2f}',fontsize=8.5,labelpad=1.5)
    if j>0: ax.set_yticklabels([])
    else: ax.set_ylabel('vibration (mm/s)',fontsize=8.5)
NUM['nu_panels']=HN
VC={}
ks=[2,3,5,10,20,40,80,150]; VC['k']=[float(roc_auc_score(Y,-LocalOutlierFactor(n_neighbors=k).fit(Z).negative_outlier_factor_)) for k in ks]
ax=fig.add_subplot(gs[1,0]); ax.plot(ks,VC['k'],'-o',color=BLUE,ms=3.2,lw=1.2); ax.set_xscale('log'); ax.set_xlabel('LOF: $k$ (n_neighbors)',labelpad=1.5); ax.set_ylabel('ROC AUC'); ax.set_ylim(0.55,1.02); ax.minorticks_off(); ax.set_xticks([2,10,40,150]); ax.set_xticklabels(['2','10','40','150']); grid(ax)
ms=[8,16,32,64,128,256,315]; mm=[];ss=[]
for q in ms:
    a=[roc_auc_score(Y,-IsolationForest(max_samples=q,random_state=r_).fit(Z).score_samples(Z)) for r_ in range(10)]; mm.append(np.mean(a)); ss.append(np.std(a))
VC['psi']=[mm,ss]
ax=fig.add_subplot(gs[1,1]); ax.errorbar(ms,mm,yerr=ss,fmt='-s',color=OR,ms=3.2,lw=1.2,capsize=2); ax.set_xscale('log'); ax.set_xlabel('iForest: $\\psi$ (max_samples)',labelpad=1.5); ax.set_ylim(0.55,1.02); ax.set_yticklabels([]); ax.minorticks_off(); ax.set_xticks([8,32,128,315]); ax.set_xticklabels(['8','32','128','315']); grid(ax)
gs_=[0.01,0.03,0.1,0.3,1,3,10,30]; VC['gamma']=[float(roc_auc_score(Y,-OneClassSVM(nu=0.05,gamma=g_).fit(Z).score_samples(Z))) for g_ in gs_]
ax=fig.add_subplot(gs[1,2]); ax.plot(gs_,VC['gamma'],'-^',color=PUR,ms=3.2,lw=1.2); ax.axvline(0.5,color=GREEN,lw=0.9,ls='--'); ax.set_xscale('log'); ax.set_xlabel('OneClassSVM: $\\gamma$',labelpad=1.5); ax.set_ylim(0.55,1.02); ax.set_yticklabels([]); ax.minorticks_off(); ax.set_xticks([0.01,0.1,1,10]); ax.set_xticklabels(['0.01','0.1','1','10']); grid(ax)
NUM['valcurves']=VC
sv(fig,'f487.png',470)

# ======== 48.11-1 case study: ROC curves and repeated draws
CR=json.load(open('verify/case_repeats.json')); R_=CR['R']; names=list(R_)
fig,axs=plt.subplots(1,2,figsize=(5.0,2.1),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; sty={'IsolationForest':(BLUE,'-'),'LocalOutlierFactor':(OR,'--'),'OneClassSVM':(GREEN,'-.'),'EllipticEnvelope':(RED,':')}
for k,(f_,t_) in CR['roc'].items(): ax.plot(f_,t_,color=sty[k][0],ls=sty[k][1],lw=1.4,label=k)
ax.plot([0,1],[0,1],color=GREY,lw=0.7); ax.set_xlabel('false alarm rate'); ax.set_ylabel('share of malignant caught'); ax.legend(loc='lower right',handlelength=2.0,labelspacing=0.2,fontsize=7.5); grid(ax)
ax=axs[1]; yy=np.arange(len(names))[::-1]
for off,key,mk,col,lab in ((0.15,'nov_auc','o',BLUE,'novelty'),(-0.15,'out_auc','s',OR,'outlier mode')):
    ax.errorbar([np.mean(R_[k][key]) for k in names],yy+off,xerr=[np.std(R_[k][key]) for k in names],fmt=mk,color=col,ms=4,capsize=2,lw=1,label=lab)
ax.set_yticks(yy); ax.set_yticklabels(['IsolationForest','LOF','OneClassSVM','EllipticEnvelope'],fontsize=8); ax.set_xlabel('ROC AUC, mean ± sd of 20 draws'); ax.set_xlim(0.8,1.0); ax.legend(loc='lower left',handletextpad=0.2,labelspacing=0.2,fontsize=7.5); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f488.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})

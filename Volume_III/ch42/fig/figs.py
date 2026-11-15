# All figures of Chapter 42. Run: python3 fig/figs.py  (PNGs into ch42/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.datasets import load_digits, load_wine, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
LO='#FDEBD0'
# ---- the five fish
FX=np.array([[27,4],[29,6],[30,6],[31,8],[33,6]],float); FMU=FX.mean(axis=0); FC=FX-FMU
fp=PCA(2).fit(FX); W1,W2=fp.components_; LAM=fp.explained_variance_; NEW=np.array([32.,5.])
NUM['fish']={'lam':LAM.tolist(),'angle_deg':float(np.degrees(np.arctan2(W1[1],W1[0])))}

# ---- 42.0-1 opener: the fish, the axes, the new coordinates
fig,axs=plt.subplots(1,2,figsize=(5.4,2.45),dpi=240,gridspec_kw={'width_ratios':[1.15,1]})
ax=axs[0]
for p in FX:
    z=(p-FMU)@W1; q=FMU+z*W1; ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=1.1,zorder=3)
ax.plot(*np.c_[FMU-3.6*W1,FMU+3.6*W1],color=GREEN,lw=1.4,zorder=2)
for w_,l_,ls in ((W1,LAM[0],'-'),(W2,LAM[1],'-')):
    arr(ax,FMU,FMU+np.sqrt(l_)*w_*1.0,c=GREEN,lw=2.2,ms=9)
ax.scatter(FX[:,0],FX[:,1],s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5); ax.scatter(*FMU,s=40,marker='+',color=NAVY,lw=1.3,zorder=6)
ax.text(33.45,7.75,'PC 1',color=GREEN,fontsize=9,weight='bold',ha='left',va='center')
ax.text(FMU[0]+1.2*W2[0]-0.15,FMU[1]+1.2*W2[1]+0.15,'PC 2',color=GREEN,fontsize=9,weight='bold',ha='right')
ax.set_aspect('equal'); ax.set_xlim(25.5,34.5); ax.set_ylim(2.2,9.2); ax.set_xlabel('length (cm)'); ax.set_ylabel('weight (100 g)'); ax.set_xticks([26,28,30,32,34]); ax.set_title('data, mean, and the two axes',fontsize=9,color=NAVY)
ax=axs[1]; Z=fp.transform(FX)
ax.axhline(0,color=GREY,lw=0.6); ax.axvline(0,color=GREY,lw=0.6)
ax.scatter(Z[:,0],Z[:,1],s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
for k,(a,b) in enumerate(Z): ax.text(a+0.18,b+0.18,str(k+1),fontsize=8.5,color=GREY)
ax.set_aspect('equal'); ax.set_xlim(-4.6,4.0); ax.set_ylim(-2.6,2.6); ax.set_xlabel('score on PC 1'); ax.set_ylabel('score on PC 2'); ax.set_title('the same fish, new coordinates',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=0.8); sv(fig,'f420.png',470)

# ---- 42.1-1 the fish, numbered, with a sixth fish
fig,ax=plt.subplots(figsize=(3.6,2.45),dpi=240)
ax.scatter(FX[:,0],FX[:,1],s=42,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
for k,(a,b) in enumerate(FX): ax.text(a+0.18,b+0.18,str(k+1),fontsize=8.5,color=GREY)
ax.scatter(*NEW,s=75,marker='*',color=PUR,edgecolor='white',lw=0.6,zorder=6); ax.text(NEW[0]+0.3,NEW[1]-0.1,'new fish F',fontsize=8.5,color=PUR,va='center')
ax.scatter(*FMU,s=50,marker='+',color=NAVY,lw=1.4,zorder=6); ax.text(FMU[0]+0.25,FMU[1]-0.55,'mean (30, 6)',fontsize=8.5,color=NAVY)
ax.axvline(FMU[0],color=GRID,lw=0.8,ls='--'); ax.axhline(FMU[1],color=GRID,lw=0.8,ls='--')
ax.set_xlim(26,35); ax.set_ylim(3,9); ax.set_xlabel('length (cm)'); ax.set_ylabel('weight (100 g)'); grid(ax)
sv(fig,'f421.png',340)

# ---- 42.2-1 variance and error depend on the direction
th=np.radians(np.linspace(-90,90,361)); S2=np.cov(FX.T)
var_th=np.array([np.array([np.cos(t),np.sin(t)])@S2@np.array([np.cos(t),np.sin(t)]) for t in th]); tot=np.trace(S2)
NUM['dir']={'total':float(tot),'var_length':float(S2[0,0]),'var_weight':float(S2[1,1]),'var_pc1':float(LAM[0]),'min_var':float(var_th.min())}
fig,axs=plt.subplots(1,3,figsize=(5.4,2.0),dpi=240,gridspec_kw={'width_ratios':[1,1,1.12]})
for ax,(w,ttl) in zip(axs[:2],((np.array([1.,0.]),'direction: length axis'),(W1,'direction: PC 1'))):
    for p in FC:
        z=p@w; q=z*w; ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=1.0,zorder=3)
    ax.plot(*np.c_[-4.2*w,4.2*w],color=GREEN,lw=1.4,zorder=2)
    ax.scatter(FC[:,0],FC[:,1],s=22,color=BLUE,edgecolor='white',lw=0.6,zorder=5)
    zz=FC@w; ax.scatter(zz*w[0],zz*w[1],s=16,color=GREEN,zorder=4)
    ax.set_aspect('equal'); ax.set_xlim(-4.6,4.6); ax.set_ylim(-3.4,3.4); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=8.5,color=NAVY)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    v=float(w@S2@w); ax.text(0,-3.2,f'variance {v:.1f}   error {tot-v:.1f}',fontsize=8.5,color=NAVY,ha='center',va='bottom')
ax=axs[2]; deg=np.degrees(th)
ax.plot(deg,var_th,color=GREEN,lw=1.6,label='variance kept'); ax.plot(deg,tot-var_th,color=RED,lw=1.4,ls='--',label='error left')
ax.axvline(NUM['fish']['angle_deg'],color=PUR,lw=1.0,ls=':'); ax.axhline(tot,color=GREY,lw=0.6,ls=':')
ax.set_xlabel('angle of the direction (degrees)'); ax.set_xlim(-90,90); ax.set_xticks([-90,-45,0,45,90]); ax.set_ylim(0,7.9); ax.legend(loc='lower center',fontsize=8,handlelength=1.6,borderaxespad=0.2,ncol=2,bbox_to_anchor=(0.5,0.99),columnspacing=0.8,handletextpad=0.4); grid(ax)
ax.text(NUM['fish']['angle_deg']+4,7.35,'PC 1',color=PUR,fontsize=8.5,ha='left',va='center'); ax.text(-88,tot+0.15,'total 7',fontsize=8,color=GREY,va='bottom'); ax.set_ylim(0,7.9)
fig.tight_layout(w_pad=0.5); sv(fig,'f422.png',470)


# ---- digits (train/test split of the book)
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
dp=PCA().fit(Xtr); cum=np.cumsum(dp.explained_variance_ratio_)
K_F={f:int(PCA(f).fit(Xtr).n_components_) for f in (0.8,0.9,0.95,0.99)}; K_MLE=int(PCA('mle').fit(Xtr).n_components_)
NUM['digits']={'k_frac':K_F,'mle':K_MLE,'cum':[round(float(c),4) for c in cum[:25]]}

# ---- 42.5-1 scree and cumulative plot
fig,axs=plt.subplots(1,2,figsize=(5.4,2.2),dpi=240)
ax=axs[0]; ax.bar(np.arange(1,31),dp.explained_variance_ratio_[:30],color=BLUE,width=0.75); ax.set_xlabel('component number $j$'); ax.set_ylabel('share of variance $r_j$'); ax.set_xlim(0,31); grid(ax); ax.set_title('scree plot',fontsize=9,color=NAVY)
ax=axs[1]; ax.plot(np.arange(1,65),cum,color=GREEN,lw=1.6)
for f,k in K_F.items():
    ax.plot([0,k],[f,f],color=GRID,lw=0.8,zorder=1); ax.plot([k,k],[0.5,f],color=GRID,lw=0.8,zorder=1); ax.scatter([k],[f],s=26,color=PUR,zorder=5)
    ax.text(k+1.5,f-0.045,f'{k}',color=PUR,fontsize=8.5,ha='left',va='center')
ax.set_xlim(0,64); ax.set_ylim(0.5,1.01); ax.set_xlabel('number of components $k$'); ax.set_ylabel('cumulative share'); ax.set_yticks([0.5,0.8,0.9,0.95,0.99]); ax.set_yticklabels(['0.5','0.8','0.9','0.95','0.99']); grid(ax); ax.set_title('cumulative share',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig,'f424.png',470)

# ---- 42.5-2 whitening
rng=np.random.default_rng(3); C0=np.array([[3.0,1.5],[1.5,1.0]]); Xw=rng.multivariate_normal([3,2],C0,300)
pw=PCA(2).fit(Xw); Zp=pw.transform(Xw); pwh=PCA(2,whiten=True).fit(Xw); Zw=pwh.transform(Xw)
NUM['whiten']={'var_raw':np.var(Xw,axis=0,ddof=1).round(3).tolist(),'var_pc':np.var(Zp,axis=0,ddof=1).round(3).tolist(),'var_w':np.var(Zw,axis=0,ddof=1).round(3).tolist(),'cov_raw':np.cov(Xw.T).round(3).tolist(),'lam':pw.explained_variance_.round(3).tolist()}
def ell(ax,cov,mean,col,nstd=(1,2)):
    w,v=np.linalg.eigh(cov); ang=np.degrees(np.arctan2(v[1,1],v[0,1]))
    from matplotlib.patches import Ellipse
    for k,ns in enumerate(nstd): ax.add_patch(Ellipse(mean,2*ns*np.sqrt(w[1]),2*ns*np.sqrt(w[0]),angle=ang,fc='none',ec=col,lw=1.2,ls='-' if k==0 else '--',zorder=4))
fig,axs=plt.subplots(1,3,figsize=(5.4,1.95),dpi=240)
for ax,(D_,ttl,lim) in zip(axs,((Xw,'raw features',(-4,10,-5,9)),(Zp,'PCA scores',(-7,7,-7,7)),(Zw,'whitened scores',(-7,7,-7,7)))):
    ax.scatter(D_[:,0],D_[:,1],s=6,color=BLUE,lw=0,alpha=0.8,zorder=3); ell(ax,np.cov(D_.T),D_.mean(axis=0),GREEN)
    ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5); ax.set_aspect('equal'); ax.set_xlim(lim[0],lim[1]); ax.set_ylim(lim[2],lim[3]); ax.set_title(ttl,fontsize=9,color=NAVY)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
v1=NUM['whiten']
axs[0].text(0.5,-0.06,'variances %.1f and %.1f'%tuple(v1['var_raw']),transform=axs[0].transAxes,ha='center',va='top',fontsize=8.5,color=NAVY)
axs[1].text(0.5,-0.06,'variances %.1f and %.1f'%tuple(v1['var_pc']),transform=axs[1].transAxes,ha='center',va='top',fontsize=8.5,color=NAVY)
axs[2].text(0.5,-0.06,'variances %.1f and %.1f'%tuple(v1['var_w']),transform=axs[2].transAxes,ha='center',va='top',fontsize=8.5,color=NAVY)
fig.tight_layout(w_pad=0.3); sv(fig,'f425.png',470)


# ---- 42.7-1 the 'auto' rule for svd_solver (from the source of release 1.9.1)
fig,ax=plt.subplots(figsize=(5.0,2.75),dpi=240); ax.set_xlim(0,9.2); ax.set_ylim(0,5.8); ax.axis('off')
def bx(x,y,w,h,t,fc,ec,fs=8.6,bold=False): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=fs,bold=bold,r=0.1)
Q=[('sparse X ?',5.0),('p ≤ 1000 and n ≥ 10 p ?',3.85),("max(n, p) ≤ 500 or k = 'mle' ?",2.7),('1 ≤ k < 0.8 · min(n, p) ?',1.55)]
R_=[('arpack','#E3F4EA'),('covariance_eigh','#E3F4EA'),('full','#E3F4EA'),('randomized','#E3F4EA')]
for (t,y),(r_,fc) in zip(Q,R_):
    bx(0.2,y,4.6,0.7,t,'white',BLUE,fs=9)
    ax.text(4.95,y+0.5,'yes',fontsize=8.5,color=GREEN,ha='left',va='center'); arr(ax,(4.8,y+0.35),(6.0,y+0.35),c=GREEN,lw=1.2,ms=7)
    bx(6.0,y,2.9,0.7,r_,fc,GREEN,fs=9,bold=True)
for k in range(3):
    y0=Q[k][1]; ax.text(2.58,y0-0.2,'no',fontsize=8.5,color=RED,ha='left',va='center'); arr(ax,(2.5,y0),(2.5,Q[k+1][1]+0.7),c=RED,lw=1.2,ms=7)
ax.text(2.58,Q[3][1]-0.2,'no',fontsize=8.5,color=RED,ha='left',va='center'); arr(ax,(2.5,Q[3][1]),(2.5,0.75),c=RED,lw=1.2,ms=7)
bx(1.0,0.05,3.0,0.7,'full','#E3F4EA',GREEN,fs=9,bold=True)
fig.subplots_adjust(0,0,1,1); sv(fig,'f426.png',440)

# ---- 42.10-1 reconstructions of test digits with k components
idx=[int(np.where(yte==d)[0][0]) for d in (3,8,5)]; KS=[1,2,5,10,20,40]
fig,axs=plt.subplots(3,7,figsize=(5.4,2.55),dpi=240); NUM['recon']={}
for j,k in enumerate([None]+KS):
    if k is None: rec=Xte; ttl='original'
    else:
        pk=PCA(k).fit(Xtr); rec=pk.inverse_transform(pk.transform(Xte)); ttl=f'k = {k}'
        NUM['recon'][str(k)]={'mse_pixel':round(float(((rec-Xte)**2).mean()),3),'mse_total':round(float(((rec-Xte)**2).sum(axis=1).mean()),3)}
    for r_,ix in enumerate(idx):
        a=axs[r_,j]; a.imshow(rec[ix].reshape(8,8),cmap='gray_r',vmin=0,vmax=16,interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values(): sp_.set_visible(True); sp_.set_color(GRID)
        if r_==0: a.set_title(ttl,fontsize=8.6,color=NAVY if k is not None else GREY,pad=2)
        if r_==2 and k is not None: a.set_xlabel('%.1f'%NUM['recon'][str(k)]['mse_total'],fontsize=8.5,color=RED,labelpad=2)
axs[2,0].set_xlabel('squared error',fontsize=8.5,color=RED,labelpad=2)
fig.subplots_adjust(left=0.05,right=0.995,top=0.9,bottom=0.1,wspace=0.06,hspace=0.06); sv(fig,'f427.png',470)

# ---- 42.10-2 validation curve: accuracy of PCA(k) + 5-NN
KL=[1,2,3,4,5,6,8,10,12,15,20,25,30,40,50,64]; cvm=[];cvs=[];cvw=[]
for k in KL:
    sc=cross_val_score(make_pipeline(PCA(k),KNeighborsClassifier(5)),Xtr,ytr,cv=5); cvm.append(sc.mean()); cvs.append(sc.std())
    cvw.append(cross_val_score(make_pipeline(PCA(k,whiten=True),KNeighborsClassifier(5)),Xtr,ytr,cv=5).mean())
raw_cv=float(cross_val_score(KNeighborsClassifier(5),Xtr,ytr,cv=5).mean()); cvm=np.array(cvm); cvs=np.array(cvs)
NUM['valcurve']={'k':KL,'cv':[round(float(v),4) for v in cvm],'cv_whiten':[round(float(v),4) for v in cvw],'raw_cv':round(raw_cv,4),'best_k':int(KL[int(np.argmax(cvm))]),'best':round(float(cvm.max()),4)}
fig,ax=plt.subplots(figsize=(4.4,2.3),dpi=240)
ax.fill_between(KL,cvm-cvs,cvm+cvs,color=LGREY,alpha=0.8,lw=0); ax.plot(KL,cvm,'-o',color=BLUE,ms=3.5,lw=1.5,label='PCA, then 5-NN'); ax.plot(KL,cvw,'--s',color=OR,ms=3.2,lw=1.2,label='with whiten=True')
ax.axhline(raw_cv,color=GREEN,lw=1.0,ls=':'); ax.text(63,raw_cv+0.012,'5-NN on the 64 pixels',color=GREEN,fontsize=8.5,ha='right',va='bottom')
ax.axvline(K_F[0.95],color=PUR,lw=1.0,ls='--'); ax.text(K_F[0.95]+1,0.6,'95 % of the\nvariance',color=PUR,fontsize=8.5,ha='left',va='center')
ax.set_xlabel('number of components $k$'); ax.set_ylabel('5-fold validation accuracy'); ax.set_ylim(0.1,1.1); ax.set_xlim(0,65); grid(ax); ax.legend(loc='lower left',fontsize=8.5,handlelength=1.8,bbox_to_anchor=(0.06,0.02))
fig.tight_layout(); sv(fig,'f428.png',390)


# ---- 42.11-1 kernel PCA on two circles
CX,CY=make_circles(n_samples=300,factor=0.3,noise=0.05,random_state=0); Ctr,Cte,cytr,cyte=train_test_split(CX,CY,test_size=0.33,random_state=0,stratify=CY)
def cpts(ax,Z,y,s=9):
    for cl,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cl; ax.scatter(Z[m,0],Z[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.3,zorder=5)
fig,axs=plt.subplots(2,3,figsize=(5.4,3.1),dpi=240); NUM['kpca']={}
cpts(axs[0,0],CX,CY); axs[0,0].set_title('the data',fontsize=9,color=NAVY); axs[0,0].set_aspect('equal')
lp=PCA(2).fit(Ctr); cpts(axs[0,1],lp.transform(CX),CY); axs[0,1].set_title('linear PCA',fontsize=9,color=NAVY)
for ax,g in zip((axs[0,2],axs[1,0],axs[1,1],axs[1,2]),(0.5,2,10,50)):
    kp=KernelPCA(2,kernel='rbf',gamma=g,random_state=0).fit(Ctr); Z=kp.transform(CX); cpts(ax,Z,CY); ax.set_title(f'RBF, $\\gamma$ = {g}',fontsize=9,color=PUR)
    NUM['kpca'][str(g)]={'eig':kp.eigenvalues_.round(2).tolist()}
for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([])
axs[1,0].set_xlabel('score 1',fontsize=8.5); axs[1,0].set_ylabel('score 2',fontsize=8.5)
for ax in (axs[0,1],axs[0,2],axs[1,1],axs[1,2]): ax.set_xlabel('')
fig.tight_layout(w_pad=0.5,h_pad=0.6); sv(fig,'f429.png',470)

# ---- 42.12-1 loadings of PCA and SparsePCA on wine
wine=load_wine(); WX=StandardScaler().fit_transform(wine.data); wn=[n.replace('_',' ') for n in wine.feature_names]
wn=[n.replace('od280/od315 of diluted wines','OD280/OD315').replace('nonflavanoid phenols','nonflav. phenols').replace('proanthocyanins','proanthocyan.').replace('alcalinity of ash','alcalinity of ash') for n in wn]
wp=PCA(3).fit(WX); ws=SparsePCA(3,alpha=5,random_state=0).fit(WX)
fig,axs=plt.subplots(1,2,figsize=(5.4,2.65),dpi=240,sharey=True)
for ax,(M,ttl) in zip(axs,((wp.components_,'PCA'),(ws.components_,'SparsePCA, $\\alpha$ = 5'))):
    im=ax.imshow(M.T,cmap='RdBu_r',vmin=-0.6,vmax=0.6,aspect='auto'); ax.set_xticks([0,1,2]); ax.set_xticklabels(['PC 1','PC 2','PC 3'],fontsize=8.5); ax.set_yticks(range(13)); ax.set_yticklabels(wn,fontsize=8.6); ax.set_title(ttl,fontsize=9,color=NAVY)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
    ax.tick_params(length=0)
    for (r_,c_),v in np.ndenumerate(M.T):
        if abs(v)<1e-12: ax.text(c_,r_,'0',ha='center',va='center',fontsize=8.6,color=GREY)
fig.tight_layout(w_pad=0.6); sv(fig,'f4210.png',450)
NUM['sparse_nz']=(np.abs(ws.components_)>1e-12).sum(axis=1).tolist()

# ---- 42.13-1 centring: PCA against TruncatedSVD
rng=np.random.default_rng(5); T=rng.multivariate_normal([0,0],[[1.6,-0.9],[-0.9,0.8]],120)+np.array([6.0,3.5])
tp=PCA(1).fit(T); ts=TruncatedSVD(1,algorithm='arpack').fit(T); NUM['trunc']={'cos':float(abs(tp.components_[0]@ts.components_[0]))}
fig,axs=plt.subplots(1,2,figsize=(4.8,2.15),dpi=240,sharex=True,sharey=True)
for ax,(v,mu_,ttl,col) in zip(axs,((tp.components_[0],tp.mean_,'PCA: line through the mean',GREEN),(ts.components_[0],np.zeros(2),'TruncatedSVD: through 0',PUR))):
    ax.scatter(T[:,0],T[:,1],s=7,color=BLUE,lw=0,alpha=0.8,zorder=3); ax.plot(*np.c_[mu_-9*v,mu_+9*v],color=col,lw=1.8,zorder=4)
    ax.scatter(0,0,s=40,marker='+',color=NAVY,lw=1.3,zorder=6); ax.set_xlim(-1.5,10); ax.set_ylim(-1.5,7.5); ax.set_title(ttl,fontsize=8.6,color=NAVY); ax.set_aspect('equal')
    ax.set_xticks([0,4,8]); ax.set_yticks([0,3,6])
axs[0].set_ylabel('feature 2'); fig.supxlabel('feature 1',fontsize=9,y=0.02)
axs[0].text(0.3,-1.1,'origin',fontsize=8.5,color=NAVY,va='bottom')
fig.tight_layout(w_pad=0.6); sv(fig,'f4211.png',390)


# ---- 42.14-1 digits: the plane of the first two components and the first ten eigendigits
dp2=PCA(2).fit(Xtr); Zd=dp2.transform(Xtr)
import matplotlib.gridspec as gs
fig=plt.figure(figsize=(5.4,2.6),dpi=240)
axl=fig.add_axes([0.095,0.17,0.46,0.80]); cols=plt.cm.tab10(np.arange(10))
for d in range(10):
    m=np.where(ytr==d)[0][:40]
    for ix in m: axl.text(Zd[ix,0],Zd[ix,1],str(d),color=cols[d],fontsize=8.4,ha='center',va='center',weight='bold')
axl.set_xlim(-30,32); axl.set_ylim(-31,33); axl.set_xlabel('score on PC 1'); axl.set_ylabel('score on PC 2'); axl.set_aspect('equal')
for r_ in range(2):
    for c_ in range(4):
        j=r_*4+c_; x0=0.585+c_*0.1045; y0=0.52-r_*0.355
        a=fig.add_axes([x0,y0,0.098,0.39]); v=dp.components_[j].reshape(8,8); a.imshow(v,cmap='RdBu_r',vmin=-0.5,vmax=0.5,interpolation='nearest'); a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values(): sp_.set_visible(False)
        a.set_title(f'PC {j+1}',fontsize=8.5,color=NAVY,pad=2)
sv(fig,'f4212.png',470)

# ---- 42.14-2 wine biplot
wp2=PCA(2).fit(WX); Zw2=wp2.transform(WX); lab=wine.target
fig,ax=plt.subplots(figsize=(4.3,2.8),dpi=240)
for cl,col,mk in ((0,BLUE,'o'),(1,OR,'s'),(2,TEAL,'^')):
    m=lab==cl; ax.scatter(Zw2[m,0],Zw2[m,1],s=16,color=col,marker=mk,edgecolor='white',lw=0.4,zorder=4,label=wine.target_names[cl])
Ld_=wp2.components_.T*np.sqrt(wp2.explained_variance_)*2.4
import matplotlib.patheffects as pe
lbl={'alcohol':(0.80,2.12,'left','alcohol'),'malic_acid':(-1.35,1.05,'right','malic acid'),'alcalinity_of_ash':(-1.35,-0.28,'right','alcalinity'),
 'total_phenols':(2.25,0.55,'left','total phenols'),'flavanoids':(2.35,-0.22,'left','flavanoids'),'nonflavanoid_phenols':(-1.65,0.42,'right','nonflav. phenols'),
 'color_intensity':(-0.55,2.2,'right','colour intensity'),'hue':(1.65,-1.3,'left','hue'),'od280/od315_of_diluted_wines':(2.1,-0.8,'left','OD280/OD315'),'proline':(1.62,1.62,'left','proline')}
for k,n in enumerate(wine.feature_names):
    arr(ax,(0,0),tuple(Ld_[k]),c=NAVY,lw=0.9,ms=6,zorder=6)
    if n in lbl:
        x_,y_,ha,t_=lbl[n]; ax.text(x_,y_,t_,fontsize=8.2,color=NAVY,ha=ha,va='center',zorder=7,path_effects=[pe.withStroke(linewidth=2.2,foreground='white')])
ax.axhline(0,color=GRID,lw=0.6); ax.axvline(0,color=GRID,lw=0.6)
ax.set_xlabel('PC 1  (%.1f %% of the variance)'%(100*wp2.explained_variance_ratio_[0])); ax.set_ylabel('PC 2  (%.1f %%)'%(100*wp2.explained_variance_ratio_[1]))
ax.legend(loc='lower left',fontsize=8.2,handletextpad=0.2,borderaxespad=0.2,labelspacing=0.2,scatterpoints=1,markerscale=0.9)
ax.set_xlim(-4.4,5.6); ax.set_ylim(-4.6,4.6)
fig.tight_layout(); sv(fig,'f4213.png',390)
NUM['wine']={'ratio':wp2.explained_variance_ratio_.round(3).tolist()}

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print('figs ok')

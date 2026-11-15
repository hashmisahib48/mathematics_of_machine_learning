# Problem figures of Chapter 45. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl, GaussianRandomProjection, SparseRandomProjection
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

# 45A.0 how many dimensions
k1,k2=int(jl(500,eps=0.4)),int(jl(500,eps=0.2)); A['45A.0']={'k_eps0.4':k1,'k_eps0.2':k2,'factor':round(k2/k1,3)}
fig,ax=small(2.4,1.5); es=np.linspace(0.1,0.9,100); ax.plot(es,jl(500,eps=es),color=BLUE,lw=1.4)
for e,k in ((0.4,k1),(0.2,k2)): ax.plot(e,k,'o',color=RED,ms=4.5,zorder=5); ax.text(e+0.03,k*1.05,f'$\\varepsilon={e}$',fontsize=9,color=NAVY)
ax.set_yscale('log'); ax.set_yticks([200,500,1000,2000]); ax.set_yticklabels(['200','500','1000','2000']); ax.minorticks_off(); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('$k$ for $n=500$'); grid(ax); sv(fig,'p_45A0.png')

# 45A.1 a projection by hand
Rm=np.array([[1,0,-1,0,1,0],[0,1,0,-1,0,-1],[-1,0,0,1,0,1]],float); u=np.array([3,1,2,0,1,2.]); v=np.array([1,1,0,2,1,0.]); d=u-v
A['45A.1']={'d':d.tolist(),'Rd':(Rm@d).tolist(),'d2':float(d@d),'Rd2':float((Rm@d)@(Rm@d)),'ratio':round(float((Rm@d)@(Rm@d)/(d@d)),4),'s':3,'k':3,'entry':float(np.sqrt(3/3))}
fig,ax=small(2.4,1.3); ax.imshow(Rm,cmap='RdBu_r',vmin=-1.3,vmax=1.3,aspect='equal')
for i in range(3):
    for j in range(6): ax.text(j,i,f'{int(Rm[i,j]):d}',ha='center',va='center',fontsize=9,color=NAVY)
ax.set_xticks(range(6)); ax.set_xticklabels([f'{j+1}' for j in range(6)]); ax.set_yticks(range(3)); ax.set_yticklabels(['1','2','3']); ax.set_xlabel('feature $l$'); ax.set_ylabel('row $j$'); ax.spines[:].set_visible(False); ax.tick_params(length=0); sv(fig,'p_45A1.png')

# 45A.2 one landmark
x3=np.array([0.,1.,2.]); K3=np.exp(-0.5*(x3[:,None]-x3[None])**2); c3=K3[:,1]; Kh3=np.outer(c3,c3)/K3[1,1]
A['45A.2']={'K':K3.round(4).tolist(),'Kh':Kh3.round(4).tolist(),'K02':round(float(K3[0,2]),4),'Kh02':round(float(Kh3[0,2]),4),'Kh00':round(float(Kh3[0,0]),4),'trace_err':round(float(np.trace(K3-Kh3)),4),'rank':int(np.linalg.matrix_rank(Kh3))}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.35),dpi=240)
for ax,M,tt in zip(axs,(K3,Kh3),('exact','one landmark')):
    ax.imshow(M,cmap='Blues',vmin=0,vmax=1); ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2]); ax.set_xticklabels(['0','1','2']); ax.set_yticklabels(['0','1','2']); ax.set_title(tt,fontsize=9,color=NAVY,pad=2)
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=9,color=NAVY if M[i,j]<0.6 else 'white')
    ax.spines[:].set_visible(False); ax.tick_params(length=0)
sv(fig,'p_45A2.png')

# data for B and C
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
# 45B.1 projection and 5-NN on the raw digits
ks=[4,8,16,32]; res={'Gaussian':[],'Sparse':[]}
for k in ks:
    for nm,cls in (('Gaussian',GaussianRandomProjection),('Sparse',SparseRandomProjection)):
        a=[]
        for s in range(10):
            rp=cls(k,random_state=s).fit(Xtr); a.append(KNeighborsClassifier(5).fit(rp.transform(Xtr),ytr).score(rp.transform(Xte),yte))
        res[nm].append((round(float(np.mean(a)),4),round(float(np.std(a)),4)))
raw=float(KNeighborsClassifier(5).fit(Xtr,ytr).score(Xte,yte)); A['45B.1']={'k':ks,'res':res,'raw':round(raw,4)}
fig,ax=small(2.4,1.5)
for nm,col,mk in (('Gaussian',BLUE,'o'),('Sparse',OR,'s')): ax.errorbar(ks,[r[0] for r in res[nm]],yerr=[r[1] for r in res[nm]],fmt='-'+mk,color=col,ms=3.5,lw=1.1,capsize=2,label=nm)
ax.axhline(raw,color=GREEN,lw=1.2,ls='--'); ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.set_xlabel('$k$'); ax.set_ylabel('accuracy'); ax.legend(loc='lower right',handlelength=1.3,labelspacing=0.2); grid(ax); sv(fig,'p_45B1.png')

# 45B.0 gamma and the Gram error
Xs=Xtr/16; gs=[0.001,0.01,0.1105,1.0]; er={'RBFSampler':[],'Nystroem':[]}
for g in gs:
    Kg=rbf_kernel(Xs,gamma=g)
    for nm,cls in (('RBFSampler',RBFSampler),('Nystroem',Nystroem)):
        Z=cls(gamma=g,n_components=300,random_state=0).fit_transform(Xs); er[nm].append(round(float(np.linalg.norm(Kg-Z@Z.T)/np.linalg.norm(Kg)),4))
A['45B.0']={'gamma':gs,'err':er}
fig,ax=small(2.4,1.5)
ax.plot(gs,er['RBFSampler'],'-s',color=OR,ms=3.5,lw=1.1,label='RBFSampler'); ax.plot(gs,np.maximum(er['Nystroem'],1e-5),'-o',color=BLUE,ms=3.5,lw=1.1,label='Nystroem')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$\\gamma$'); ax.set_ylabel('relative error'); ax.legend(loc='upper left',handlelength=1.3,labelspacing=0.2); grid(ax); sv(fig,'p_45B0.png')

# 45B.2 k-means landmarks
g=0.1105; Kg=rbf_kernel(Xs,gamma=g)
def nys_err(L):
    W=rbf_kernel(L,L,gamma=g); lam,U=np.linalg.eigh(W); Wmh=(U/np.sqrt(np.clip(lam,1e-12,None)))@U.T; Z=rbf_kernel(Xs,L,gamma=g)@Wmh; return float(np.linalg.norm(Kg-Z@Z.T)/np.linalg.norm(Kg))
er_r=[];er_k=[]
for s in range(5):
    idx=np.random.RandomState(s).permutation(len(Xs))[:50]; er_r.append(nys_err(Xs[idx]))
    er_k.append(nys_err(KMeans(50,n_init=1,random_state=s).fit(Xs).cluster_centers_))
A['45B.2']={'random':r4(er_r),'kmeans':r4(er_k),'mean_random':round(float(np.mean(er_r)),4),'mean_kmeans':round(float(np.mean(er_k)),4)}
fig,ax=small(2.4,1.4)
ax.plot(np.zeros(5)+np.linspace(-0.08,0.08,5),er_r,'o',color=BLUE,ms=4); ax.plot(1+np.linspace(-0.08,0.08,5),er_k,'s',color=GREEN,ms=4)
ax.set_xlim(-0.5,1.5); ax.set_xticks([0,1]); ax.set_xticklabels(['random','k-means']); ax.set_ylabel('Gram error'); grid(ax); sv(fig,'p_45B2.png')

# 45C.1 auto and the digits
nn=len(Xtr); ka={e:int(jl(nn,eps=e)) for e in (0.5,0.3,0.1)}; A['45C.1']={'n':nn,'k':ka,'p':64}
fig,ax=small(2.4,1.5); ax.bar([0,1,2],[ka[0.5],ka[0.3],ka[0.1]],color=GREY,width=0.55); ax.axhline(64,color=RED,lw=1.4); ax.set_yscale('log'); ax.set_xticks([0,1,2]); ax.set_xticklabels(['0.5','0.3','0.1']); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('$k$ required'); ax.text(2.45,70,'$p=64$',color=RED,fontsize=9,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.0)); ax.set_ylim(30,2e4); grid(ax); sv(fig,'p_45C1.png')

# 45C.2 spiky and flat
p20=400; fl=np.ones(p20)/np.sqrt(p20); sp=np.zeros(p20); sp[0]=1.0; sq=np.sqrt(p20)
th=lambda x,s,k:np.sqrt((2+(s-3)*(x**4).sum()/(x@x)**2)/k); A['45C.2']={'p':p20,'s':round(float(sq),3),'k':10,'flat':round(float(th(fl,sq,10)),4),'spiky':round(float(th(sp,sq,10)),4),'gauss':round(float(np.sqrt(2/10)),4)}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.3),dpi=240,sharey=True)
for ax,x,tt in zip(axs,(fl,sp),('flat','spiky')):
    ax.bar(range(p20),x,color=BLUE,width=1.0); ax.set_title(tt,fontsize=9,color=NAVY,pad=2); ax.set_xticks([]); ax.set_ylim(0,1.1)
axs[0].set_ylabel('$x_l$'); sv(fig,'p_45C2.png')

A.pop('45A.0');A.pop('45B.0')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

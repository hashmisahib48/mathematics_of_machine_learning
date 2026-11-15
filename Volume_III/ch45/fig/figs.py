# All figures of Chapter 45. Run: python3 fig/figs.py  (PNGs into ch45/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import chi2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.random_projection import johnson_lindenstrauss_min_dim as jl, SparseRandomProjection, GaussianRandomProjection
from sklearn.svm import SVC, LinearSVC
from spectra import make_spectra
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# ---- data
Dg,Dy=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Dg/16,Dy,test_size=0.25,random_state=0,stratify=Dy)
GAM=1/(Xtr.shape[1]*Xtr.var()); Ktr=rbf_kernel(Xtr,gamma=GAM)
SX,SY=make_spectra(); Str,Ste,Sytr,Syte=train_test_split(SX,SY,test_size=0.25,random_state=0,stratify=SY)

# ======== 45.0-1 opener
fig,axs=plt.subplots(1,2,figsize=(5.0,2.35),dpi=240)
rng=np.random.RandomState(5); I,J=rng.randint(0,len(Str),(2,1500)); m=I!=J; I,J=I[m],J[m]
d2=((Str[I]-Str[J])**2).sum(1); R=GaussianRandomProjection(10,random_state=3).fit(Str); Zs=R.transform(Str); q2=((Zs[I]-Zs[J])**2).sum(1)
ax=axs[0]; d2m,q2m=d2/1e6,q2/1e6; ax.scatter(d2m,q2m,s=4,color=BLUE,alpha=0.5,lw=0); mx=max(d2m.max(),q2m.max())*1.03; ax.plot([0,mx],[0,mx],color=GREEN,lw=1.3); ax.set_xlim(0,mx); ax.set_ylim(0,mx); ax.set_aspect('equal')
ax.set_xlabel('squared distance, $p=10\\,000$ (millions)'); ax.set_ylabel('squared distance, $k=10$'); ax.set_title('random projection',fontsize=9,color=NAVY); grid(ax)
NUM['open_left']={'corr':float(np.corrcoef(d2,q2)[0,1]),'ratio_mean':float((q2/d2).mean()),'ratio_sd':float((q2/d2).std())}
ax=axs[1]; I2,J2=rng.randint(0,len(Xtr),(2,2500)); kx=Ktr[I2,J2]
for cls,mk,col,lab in ((RBFSampler,'s',OR,'RBFSampler'),(Nystroem,'o',BLUE,'Nystroem')):
    Z=cls(gamma=GAM,n_components=100,random_state=0).fit_transform(Xtr); ax.scatter(kx,(Z[I2]*Z[J2]).sum(1),s=5,marker=mk,color=col,alpha=0.45,lw=0,label=lab)
ax.plot([0,1],[0,1],color=GREEN,lw=1.3); ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.4,1.4); ax.set_xlabel('exact $k(\\mathbf{x},\\mathbf{z})$'); ax.set_ylabel('approximate'); ax.set_title('kernel approximation, 100 features',fontsize=9,color=NAVY)
leg=ax.legend(loc='upper left',markerscale=2.5,handletextpad=0.1,borderaxespad=0.2); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f450.png',470)

# ======== 45.2-1 intuition: shadows and the spread of the ratio
fig,axs=plt.subplots(1,2,figsize=(5.0,2.3),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; ax.set_aspect('equal'); ax.axis('off'); u=np.array([1.0,0.0]); rr=np.random.RandomState(2)
ang=np.sort(rr.uniform(0,np.pi,6))
for a in ang:
    d=np.array([np.cos(a),np.sin(a)]); ax.plot([-1.25*d[0],1.25*d[0]],[-1.25*d[1],1.25*d[1]],color=GRID,lw=0.9,zorder=1)
    s=u@d; ax.plot([0,s*d[0]],[0,s*d[1]],color=GREEN,lw=2.4,zorder=3,solid_capstyle='butt'); ax.plot([u[0],s*d[0]],[u[1],s*d[1]],color=GREY,lw=0.6,ls=':',zorder=2)
arr(ax,(0,0),(1,0),c=BLUE,lw=2.0,ms=9); ax.text(1.05,0.03,'$\\mathbf{u}$',color=BLUE,fontsize=10,ha='left',va='bottom')
ax.set_xlim(-1.3,1.45); ax.set_ylim(-1.15,1.15); ax.set_title('one vector, six random lines:\nthe shadow (green) varies',fontsize=8.5,color=NAVY)
ax=axs[1]; p_=100; x0=np.random.RandomState(0).normal(size=p_); x0/=np.linalg.norm(x0); cols={1:RED,4:OR,16:GREEN,64:BLUE}
tt=np.linspace(0,3.2,400); S={}
for k,c_ in cols.items():
    rr=np.random.RandomState(k); q=np.array([ (np.linalg.norm(rr.normal(scale=1/np.sqrt(k),size=(k,p_))@x0))**2 for _ in range(4000)]); S[k]=float(q.std())
    ax.hist(q,bins=np.linspace(0,3.2,44),density=True,histtype='stepfilled',color=c_,alpha=0.22,lw=0)
    ax.plot(tt,chi2.pdf(tt*k,k)*k,color=c_,lw=1.3,label=f'$k={k}$')
ax.axvline(1,color=NAVY,lw=0.8,ls=':'); ax.set_xlim(0,3.2); ax.set_ylim(0,3.4); ax.set_xlabel('$q=\\Vert\\mathbf{R}\\mathbf{u}\\Vert^2/\\Vert\\mathbf{u}\\Vert^2$'); ax.set_ylabel('density'); ax.legend(loc='upper right',handlelength=1.2,ncol=2,columnspacing=0.8); grid(ax)
NUM['hist_sd']=S
fig.tight_layout(w_pad=0.8); sv(fig,'f451.png',470)

# ======== 45.4-1 random Fourier features and Nystroem on a line
fig,axs=plt.subplots(1,2,figsize=(5.0,2.35),dpi=240,sharey=True)
t=np.linspace(-4,4,201)[:,None]; g=0.5; kt=np.exp(-g*t[:,0]**2)
ax=axs[0]
for D,c_ in ((2,RED),(10,OR),(100,BLUE)):
    r_=RBFSampler(gamma=g,n_components=D,random_state=1).fit(t); Fz=r_.transform(t); f0=r_.transform(np.array([[0.0]]))[0]; ax.plot(t[:,0],Fz@f0,color=c_,lw=1.1+0.0*D,label=f'$D={D}$')
ax.plot(t[:,0],kt,color=GREEN,lw=2.0,ls='--',label='exact'); ax.set_xlabel('$z$'); ax.set_ylabel('$k(0,z)$ and estimate'); ax.set_title('random Fourier features',fontsize=9,color=NAVY); ax.legend(loc='upper right',handlelength=1.4,labelspacing=0.2,fontsize=8); ax.set_ylim(-0.75,1.45); grid(ax)
ax=axs[1]; pool=np.linspace(-4,4,41)[:,None]
for m_,c_ in ((2,RED),(4,OR),(8,BLUE)):
    ny=Nystroem(gamma=g,n_components=m_,random_state=4).fit(pool); Fz=ny.transform(t); f0=ny.transform(np.array([[0.0]]))[0]; ax.plot(t[:,0],Fz@f0,color=c_,lw=1.3,label=f'$m={m_}$')
    if m_==4: ax.plot(ny.components_[:,0],np.full(m_,-0.62),'^',color=c_,ms=4.5)
ax.plot(t[:,0],kt,color=GREEN,lw=2.0,ls='--',label='exact'); ax.set_xlabel('$z$'); ax.set_title('Nyström, random landmarks',fontsize=9,color=NAVY); ax.legend(loc='upper right',handlelength=1.4,labelspacing=0.2,fontsize=8); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f453.png',470)

# ======== 45.6-1 flow chart
fig,ax=plt.subplots(figsize=(5.0,2.5),dpi=240); ax.set_xlim(0,10); ax.set_ylim(0,6.6); ax.axis('off')
def row(y,name,items):
    ax.text(0.0,y+0.95,name,fontsize=8.5,color=RED,weight='bold',va='center')
    x=0.0; gap=0.38
    for k_,(w_,txt,kind) in enumerate(items):
        fc,ec={'d':(LB,BLUE),'s':(LGREY,GREY),'o':('#E3F4EA',GREEN),'m':(PINK,RED)}[kind]
        box(ax,x,y-0.5,w_,1.0,txt,fc=fc,ec=ec,fs=8); x+=w_
        if k_<len(items)-1: arr(ax,(x,y),(x+gap,y),c=NAVY,lw=1,ms=7); x+=gap
row(5.0,'random projection',[(1.0,'data\n$\\mathbf{X}$','d'),(3.0,'draw $\\mathbf{R}$, $k\\times p$\nEq. 45.3-1, 45.3-4','s'),(2.6,'$\\mathbf{Z}=\\mathbf{X}\\mathbf{R}^{T}$\n$n\\times k$','o'),(1.3,'linear\nmodel','m')])
row(2.9,'RBFSampler',[(1.0,'data\n$\\mathbf{X}$','d'),(3.0,'draw $\\boldsymbol{\\omega}_j$, $b_j$\nEq. 45.4-3','s'),(3.4,'$\\mathbf{Z}=\\sqrt{2/D}\\,\\cos(\\mathbf{X}\\boldsymbol{\\Omega}+\\mathbf{b})$\n$n\\times D$','o'),(1.3,'linear\nmodel','m')])
row(0.8,'Nystroem',[(1.0,'data\n$\\mathbf{X}$','d'),(1.9,'landmarks\n$\\mathbf{W}=\\mathbf{K}_{II}$','s'),(1.8,'$\\mathbf{W}^{-1/2}$\nEq. 45.4-7','s'),(2.2,'$\\mathbf{Z}=\\mathbf{K}_{:,I}\\mathbf{W}^{-1/2}$\n$n\\times m$','o'),(1.0,'linear\nmodel','m')])
sv(fig,'f454.png',420)

# ======== 45.10-1 timing on make_classification (numbers from code/timing.py)
T=json.load(open('verify/timing.json')); ns=[int(k) for k in T]
fig,axs=plt.subplots(1,2,figsize=(5.0,2.35),dpi=240)
sty=[('exact SVC',RED,'-^',0),('Nystroem(300) + SGD',BLUE,'-o',1),('RBFSampler(1000) + SGD',OR,'-s',2)]
ax=axs[0]
for nm,c_,mk,j in sty:
    xs=[n for n in ns if T[str(n)][j][0]==T[str(n)][j][0]]; ax.plot(xs,[T[str(n)][j][0] for n in xs],mk,color=c_,ms=4,lw=1.3,label=nm)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('training samples $n$'); ax.set_ylabel('fit time (s)'); ax.legend(loc='lower right',handlelength=1.8,labelspacing=0.2,fontsize=8); grid(ax)
ax=axs[1]
for nm,c_,mk,j in sty:
    xs=[n for n in ns if T[str(n)][j][1]==T[str(n)][j][1]]; ax.plot(xs,[T[str(n)][j][1] for n in xs],mk,color=c_,ms=4,lw=1.3)
ax.set_xscale('log'); ax.set_xlabel('training samples $n$'); ax.set_ylabel('test accuracy'); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f457.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)
json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})

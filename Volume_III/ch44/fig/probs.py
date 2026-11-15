# Problem figures of Chapter 44. Run: python3 fig/probs.py (answers -> verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import make_s_curve, load_wine, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import ClassicalMDS, Isomap, LocallyLinearEmbedding, TSNE, trustworthiness
from sklearn.model_selection import train_test_split
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.5): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min([t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]+[99]); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*126,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 44A.1 three villages, right triangle 6-8-10
D=np.array([[0,6,8],[6,0,10],[8,10,0]],float); n=3; J=np.eye(n)-np.ones((n,n))/n; B=-0.5*J@D**2@J; w,V=np.linalg.eigh(B); w=w[::-1]; V=V[:,::-1]
cm=ClassicalMDS(2,metric='precomputed').fit(D)
A['44A.1']={'B':B.round(4).tolist(),'eig':r4(w[:2]),'Z':cm.embedding_.round(4).tolist(),'eig_third':round(float(w[2]),6)}
fig,ax=small(2.4,1.25); P=np.array([[3.6,4.8],[0,0],[10,0]]); ax.plot(*np.r_[P,P[:1]].T,color=BLUE,lw=1.2); ax.scatter(*P.T,s=22,color=BLUE,zorder=5)
for nm,p,o in zip('PQR',P,((-0.3,0.6),(-1.2,-0.4),(0.5,-0.4))): ax.text(p[0]+o[0],p[1]+o[1],nm,fontsize=9,color=NAVY)
ax.text(1.3,2.6,'6',fontsize=9,color=GREY,ha='center'); ax.text(7.2,2.8,'8',fontsize=9,color=GREY); ax.text(5,-1.5,'10',fontsize=9,color=GREY,ha='center')
ax.set_xlim(-2,11.5); ax.set_ylim(-2.3,6); ax.set_aspect('equal'); ax.axis('off'); sv(fig,'p_44A1.png')

# 44A.2 LLE weights
x=np.array([2.,1.]); nb=np.array([[0,0],[5,0.]]); C=nb-x; G=C@C.T; wv=np.linalg.solve(G,np.ones(2)); wv/=wv.sum(); rec=wv@nb
R=0.01*np.trace(G); w2=np.linalg.solve(G+R*np.eye(2),np.ones(2)); w2/=w2.sum()
A['44A.2']={'G':G.tolist(),'w':r4(wv),'rec':r4(rec),'err2':round(float(((x-rec)**2).sum()),4),'w_reg':r4(w2),'trace':float(np.trace(G))}
fig,ax=small(2.4,1.1); ax.plot(*nb.T,color=GRID,lw=1); ax.scatter(*nb.T,s=26,color=BLUE,zorder=5); ax.scatter(*x,s=50,marker='*',color=NAVY,zorder=6)
ax.text(2.2,1.2,'$\\mathbf{x}_i$',fontsize=9,color=NAVY); ax.text(0,-0.9,'(0, 0)',fontsize=9,color=NAVY,ha='center'); ax.text(5,-0.9,'(5, 0)',fontsize=9,color=NAVY,ha='center'); ax.set_xlim(-1,6); ax.set_ylim(-1.8,2.2); ax.set_aspect('equal'); ax.axis('off'); sv(fig,'p_44A2.png')

# 44A.3 perplexity
def perp(p): p=np.array(p); return float(np.exp(-(p*np.log(p)).sum()))
A['44A.3']={'a':round(perp([0.5,0.25,0.25]),4),'b':round(perp([0.25]*4),4),'c':round(perp([0.7,0.2,0.1]),4)}
fig,ax=small(2.4,0.95); ax.bar(np.arange(3)-0.2,[0.5,0.25,0.25],0.38,color=BLUE,label='(a)'); ax.bar(np.arange(3)+0.2,[0.7,0.2,0.1],0.38,color=OR,label='(c)'); ax.set_xticks(range(3)); ax.set_xticklabels(['1','2','3']); ax.set_xlabel('neighbour $j$'); ax.set_ylabel('$p_{j|i}$'); ax.legend(loc='upper right',fontsize=9); grid(ax); sv(fig,'p_44A3.png')

# 44A.4 stress
d=np.array([3,4,5.]); z=np.array([0,3,7.]); e=np.array([abs(z[0]-z[1]),abs(z[0]-z[2]),abs(z[1]-z[2])]); st=float(((d-e)**2).sum()); s1=float(np.sqrt(st/(e**2).sum()))
A['44A.4']={'embedded':e.tolist(),'stress':st,'stress1':round(s1,4)}
fig,ax=small(2.4,1.2); ax.plot([0,7],[0,0],color=GRID,lw=1); ax.scatter(z,np.zeros(3),s=30,color=BLUE,zorder=5)
for k,zz in enumerate(z): ax.text(zz,0.5,f'$z_{k+1}$ = {zz:g}',fontsize=9,color=NAVY,ha='center')
ax.set_xlim(-1,8); ax.set_ylim(-1,1.6); ax.axis('off'); sv(fig,'p_44A4.png')

# data
Xs,ts=make_s_curve(800,random_state=0)
# 44B.1 classical MDS = PCA on wine
WX=load_wine().data; WX=(WX-WX.mean(0))/WX.std(0); pc=PCA(2).fit(WX); Zp=pc.transform(WX); cmw=ClassicalMDS(2).fit(WX); sg=np.sign((Zp*cmw.embedding_).sum(0))
A['44B.1']={'max_diff_after_sign':float(f"{np.abs(Zp-cmw.embedding_*sg).max():.1e}"),'ratio_eig':r4(cmw.eigenvalues_/pc.explained_variance_),'n':len(WX)}
fig,ax=small(2.4,1.5); ax.scatter(Zp[:,0],cmw.embedding_[:,0]*sg[0],s=3,color=PUR); ax.set_xlabel('PCA score 1'); ax.set_ylabel('ClassicalMDS 1'); grid(ax); sv(fig,'p_44B1.png')

# 44B.2 Isomap components / trust on S-curve
ms=[3,5,10,30]; rows=[]
for m in ms:
    nc=int(connected_components(kneighbors_graph(Xs,m))[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); T=trustworthiness(Xs,Isomap(n_neighbors=m).fit_transform(Xs),n_neighbors=10)
    rows.append((m,nc,round(float(T),4)))
A['44B.2']=rows
fig,ax=small(2.4,1.5); ax.plot([r[0] for r in rows],[r[2] for r in rows],'-o',color=PUR,ms=3.5,lw=1.3); ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(ms); ax.set_xticklabels([str(m) for m in ms]); ax.set_xlabel('neighbours $m$'); ax.set_ylabel('T(10)'); grid(ax); sv(fig,'p_44B2.png')

# 44B.3 LLE reg
regs=[1e-6,1e-3,1e-1,1.0]; rr=[]
for rg in regs:
    l=LocallyLinearEmbedding(n_neighbors=10,reg=rg,random_state=0); Z=l.fit_transform(Xs); rr.append((rg,float(f'{l.reconstruction_error_:.3e}'),round(float(trustworthiness(Xs,Z,n_neighbors=10)),4)))
A['44B.3']=rr
fig,ax=small(2.4,1.5); ax.plot(range(4),[r[2] for r in rr],'-o',color=GREEN,ms=3.5,lw=1.3); ax.set_xticks(range(4)); ax.set_xticklabels(['$10^{-6}$','$10^{-3}$','0.1','1']); ax.set_xlabel('reg'); ax.set_ylabel('T(10)'); grid(ax); sv(fig,'p_44B3.png')

# C figures
fig,ax=small(2.4,1.5); rng=np.random.RandomState(1); Zc=np.r_[rng.normal([0,0],0.5,(40,2)),rng.normal([3,1],0.5,(40,2))]; ax.scatter(Zc[:,0],Zc[:,1],s=5,color=BLUE); ax.scatter([1.5],[0.5],marker='*',s=90,color=OR,zorder=5); ax.text(1.5,1.3,'new image',fontsize=9,color=OR,ha='center'); ax.axis('off'); sv(fig,'p_44C1.png')
fig,ax=small(2.4,1.5); rr_=np.sqrt(rng.uniform(1,4,300)); aa=rng.uniform(0,2*np.pi,300); ax.scatter(rr_*np.cos(aa),rr_*np.sin(aa),s=4,color=BLUE); ax.set_aspect('equal'); ax.axis('off'); sv(fig,'p_44C2.png')
fig,ax=small(2.4,1.5); Zb=np.r_[rng.normal([0,0],0.25,(50,2)),rng.normal([2.5,0],0.25,(50,2)),rng.normal([1.2,2],0.25,(50,2))]; ax.scatter(Zb[:,0],Zb[:,1],s=5,color=PUR); ax.axis('off'); sv(fig,'p_44C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

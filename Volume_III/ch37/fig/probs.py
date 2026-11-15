# Problem figures of Chapter 37. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.cluster.hierarchy import linkage as sl_link, dendrogram as sl_dend
from sklearn.cluster import AgglomerativeClustering as AC, FeatureAgglomeration as FA
from sklearn.datasets import load_iris, load_digits, make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from agglo_scratch import Agglo, LW
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.6): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min([t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()] or [9]); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
X7=np.array([[3,3],[1,2],[0,2],[7,4],[11,2],[5,12],[8,9]],float)

# 37A.1 average linkage on A-E
X5=X7[:5]; Z=sl_link(X5,'average'); n=5; cl={i:'ABCDE'[i] for i in range(n)}; seq=[]
for k,(a,b,h,c) in enumerate(Z):
    cl[n+k]=''.join(sorted(cl[int(a)]+cl[int(b)])); seq.append((cl[n+k],round(float(h),3)))
m=Agglo('average',2).fit(X5); assert np.allclose(m.distances_,Z[:,2])
A['37A.1']={'average':seq}
fig,ax=small(2.4,1.0)
ax.scatter(X5[:,0],X5[:,1],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=5)
for k,p in enumerate(X5): ax.text(p[0]+(-0.4 if k==2 else 0.4),p[1]+(0.3),'ABCDE'[k],fontsize=9,color=NAVY,weight='bold',ha='right' if k==2 else 'left')
ax.set_xlim(-1.5,12.5); ax.set_ylim(0.8,5.4); ax.set_xticks([0,4,8,12]); ax.set_yticks([2,4]); grid(ax); sv(fig,'p_37A1.png')

# 37A.2 Ward on four points
P4=np.array([[0,0],[2,0],[0,4],[6,4]],float); nm='PQRS'
def ss(Pm): Pm=np.atleast_2d(Pm); return float(((Pm-Pm.mean(0))**2).sum())
pair={nm[i]+nm[j]:round(0.5*float(((P4[i]-P4[j])**2).sum()),4) for i in range(4) for j in range(i+1,4)}
mu=P4[:2].mean(0); d_R=round(2*1/3*float(((P4[2]-mu)**2).sum()),4); d_S=round(2*1/3*float(((P4[3]-mu)**2).sum()),4)
mu3=P4[:3].mean(0); d_last=round(3*1/4*float(((P4[3]-mu3)**2).sum()),4)
Zp=sl_link(P4,'ward'); A['37A.2']={'pairs':pair,'PQ+R':d_R,'PQ+S':d_S,'PQR+S':d_last,'heights':[round(float(h),4) for h in Zp[:,2]],'sum_delta':round(2+d_R+d_last,4),'SS_total':round(ss(P4),4),'RS':pair['RS']}
assert abs(Zp[0,2]**2/2-pair['PQ'])<1e-9
fig,ax=small(2.4,1.15)
ax.scatter(P4[:,0],P4[:,1],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=5)
for k,p in enumerate(P4): ax.text(p[0]+0.3,p[1]+0.35,nm[k],fontsize=9,color=NAVY,weight='bold')
ax.set_xlim(-1,7.5); ax.set_ylim(-0.8,5.5); ax.set_xticks([0,2,4,6]); ax.set_yticks([0,2,4]); grid(ax); sv(fig,'p_37A2.png')

# 37A.3 a CF merge and BIRCH threshold
Pa=np.array([[1,1],[1,3]],float); Pb=np.array([[3,1],[3,3]],float)
def cf(Pm): return len(Pm),Pm.sum(0),float((Pm**2).sum())
def rad(N,LS,SS): c=LS/N; return float(np.sqrt(SS/N-c@c))
c1,c2=cf(Pa),cf(Pb); cm=(c1[0]+c2[0],c1[1]+c2[1],c1[2]+c2[2])
A['37A.3']={'CF1':[c1[0],c1[1].tolist(),c1[2]],'CF2':[c2[0],c2[1].tolist(),c2[2]],'merged':[cm[0],cm[1].tolist(),cm[2]],'R1':round(rad(*c1),4),'R2':round(rad(*c2),4),'R_merged':round(rad(*cm),4),'centroid':(cm[1]/cm[0]).tolist()}
fig,ax=small(2.4,1.05)
ax.scatter(Pa[:,0],Pa[:,1],s=30,color=BLUE,marker='o',edgecolor='white',lw=0.7,zorder=5); ax.scatter(Pb[:,0],Pb[:,1],s=30,color=OR,marker='s',edgecolor='white',lw=0.7,zorder=5)
ax.text(1.15,2,'CF$_1$',fontsize=9,color=BLUE,va='center'); ax.text(2.85,2,'CF$_2$',fontsize=9,color=OR,va='center',ha='right')
ax.set_xlim(0,4); ax.set_ylim(0.2,3.8); ax.set_xticks([0,1,2,3,4]); ax.set_yticks([1,2,3]); grid(ax); sv(fig,'p_37A3.png')

# 37A.4 cutting a chain
x1=np.array([0,0.8,2.1,4.1,6.2,11.6]).reshape(-1,1); res={}
for t in (1.5,2.1,3.0,6.0): res[str(t)]=int(AC(n_clusters=None,linkage='single',distance_threshold=t).fit(x1).n_clusters_)
Z1=sl_link(x1,'single'); A['37A.4']={'heights':[round(float(h),2) for h in Z1[:,2]],'n_clusters':res}
fig,ax=small(2.4,1.15); sl_dend(Z1,ax=ax,color_threshold=0,above_threshold_color=NAVY,no_labels=False,labels=list('abcdef')); 
for l in ax.lines: l.set_linewidth(1.3)
ax.set_ylabel('height',fontsize=9.5); ax.set_yticks([0,2,4,6]); ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x',length=0,labelsize=9); sv(fig,'p_37A4.png')

# 37B.1 iris, four linkages
Xi,yi=load_iris(return_X_y=True); A['37B.1']={L:round(float(ari(yi,AC(n_clusters=3,linkage=L).fit_predict(Xi))),3) for L in ('single','complete','average','ward')}
fig,ax=small(2.4,1.15)
for k,(col,mk) in enumerate(zip([BLUE,OR,GREEN],['o','s','^'])): ax.scatter(Xi[yi==k,2],Xi[yi==k,3],s=7,color=col,marker=mk,lw=0)
ax.set_xlabel('petal length (cm)'); ax.set_ylabel('petal width (cm)'); ax.set_xticks([2,4,6]); ax.set_yticks([0,1,2]); grid(ax); sv(fig,'p_37B1.png')

# 37B.2 centroid linkage and an inversion
Pt=np.array([[0,0],[2,0],[1,1.7]]); 
LW['centroid']=lambda i,j,k: (i/(i+j), j/(i+j), -i*j/(i+j)**2, 0)
class AggC(Agglo): pass
def centroid_heights(Xc):
    D=((Xc[:,None]-Xc[None])**2).sum(-1); D=D.astype(float); np.fill_diagonal(D,np.inf); size=np.ones(len(Xc)); alive=list(range(len(Xc))); H=[]
    while len(alive)>1:
        sub=D[np.ix_(alive,alive)]; a,b=divmod(sub.argmin(),len(alive)); i,j=alive[a],alive[b]; d=D[i,j]; H.append(float(np.sqrt(d)))
        rest=[k for k in alive if k not in (i,j)]; ai,aj,be,ga=LW['centroid'](size[i],size[j],size[rest])
        new=ai*D[i,rest]+aj*D[j,rest]+be*d; D[i,rest]=D[rest,i]=new; size[i]+=size[j]; alive.remove(j)
    return H
Hc=centroid_heights(Pt); Zc=sl_link(Pt,'centroid'); assert np.allclose(Hc,Zc[:,2])
A['37B.2']={'heights':[round(h,4) for h in Hc],'inversion':bool(Hc[1]<Hc[0])}
fig,ax=small(2.4,1.1)
ax.scatter(Pt[:,0],Pt[:,1],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=5)
for k,p in enumerate(Pt): ax.text(p[0]+0.1,p[1]-0.3 if k<2 else p[1]+0.25,'123'[k],fontsize=9,color=NAVY,weight='bold',ha='left' if k<2 else 'center')
ax.set_xlim(-0.5,2.5); ax.set_ylim(-0.6,2.4); ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2]); grid(ax); sv(fig,'p_37B2.png')

# 37B.3 pooling_func on the digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xd,yd=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.25,random_state=0,stratify=yd)
res={}
for nmf,fn in (('mean',np.mean),('max',np.max),('median',np.median)):
    f=FA(n_clusters=16,pooling_func=fn).fit(Xtr)
    res[nmf]=round(float(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(f.transform(Xtr),ytr).score(f.transform(Xte),yte)),4)
A['37B.3']=res
fig,axs=plt.subplots(1,4,figsize=(2.4,0.75),dpi=240)
for ax,k in zip(axs,[0,1,2,3]): ax.imshow(Xd[k].reshape(8,8),cmap='Greys',interpolation='nearest'); ax.axis('off')
fig.subplots_adjust(wspace=0.05,left=0,right=1,top=1,bottom=0); sv(fig,'p_37B3.png')

# 37C.1 a new sample after Ward clustering
Xn,yn=make_blobs(n_samples=90,centers=[[0,0],[4,0],[2,3.4]],cluster_std=0.6,random_state=2); mn=AC(n_clusters=3).fit(Xn)
fig,ax=small(2.4,1.2)
for c,col,mk in zip(range(3),[BLUE,OR,PUR],['o','s','^']): ax.scatter(Xn[mn.labels_==c,0],Xn[mn.labels_==c,1],s=8,color=col,marker=mk,lw=0)
ax.scatter([1.9],[1.0],s=75,marker='*',color=RED,edgecolor='white',lw=0.6,zorder=6); ax.text(2.15,1.0,'new',fontsize=9,color=RED,va='center')
ax.set_xlim(-1.5,5.5); ax.set_ylim(-1.5,4.8); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4]); grid(ax); sv(fig,'p_37C1.png')

# 37C.2 chaining
rng=np.random.default_rng(4); Ba=rng.normal([0,0],0.5,(40,2)); Bb=rng.normal([5,0],0.5,(40,2)); bridge=np.c_[np.linspace(1,4,7),rng.normal(0,0.08,7)]
Xch=np.r_[Ba,Bb,bridge]; lab_s=AC(n_clusters=2,linkage='single').fit_predict(Xch); lab_w=AC(n_clusters=2,linkage='ward').fit_predict(Xch)
A['37C.2']={'single_sizes':sorted(np.bincount(lab_s).tolist()),'ward_sizes':sorted(np.bincount(lab_w).tolist())}
fig,ax=small(2.4,1.05)
ax.scatter(Ba[:,0],Ba[:,1],s=7,color=BLUE,lw=0); ax.scatter(Bb[:,0],Bb[:,1],s=7,color=OR,marker='s',lw=0); ax.scatter(bridge[:,0],bridge[:,1],s=10,color=RED,marker='^',lw=0)
ax.set_xlim(-1.8,6.8); ax.set_ylim(-1.7,1.7); ax.set_xticks([0,2,4,6]); ax.set_yticks([-1,0,1]); grid(ax); sv(fig,'p_37C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print(json.dumps(A,indent=0)[:3000]); print(CHK)

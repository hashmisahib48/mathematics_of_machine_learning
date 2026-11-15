# All figures of Chapter 37. Run: python3 fig/figs.py  (PNGs into ch37/, numbers into verify/fig_numbers.json)
import sys,os,json,time,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage as sl_link, dendrogram as sl_dend, fcluster
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering as AC, Birch, FeatureAgglomeration as FA, KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles, make_swiss_roll, load_wine, load_digits
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
CLU=[BLUE,OR,PUR,TEAL,MAG,GREEN,GREY]; MK=['o','s','^','D','v','P','X']
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# ---- the seven finches
X7=np.array([[3,3],[1,2],[0,2],[7,4],[11,2],[5,12],[8,9]],float); NM='ABCDEFG'
Zw=sl_link(X7,'ward')                                  # heights = distances_ of sklearn
fam=np.array([0,0,0,1,1,2,2])                         # families A-C, D-E, F-G
lab3=fam.copy()                                       # Ward, K = 3, relabelled by family
assert len(set(zip(fcluster(Zw,3,'maxclust'),fam)))==3
def dendro(ax,Z,leaf_labels=None,cut=None,colors=None,lw=1.3,fs=8.5,show_leaf=True,rug=None,ymax=None):
    """own dendrogram: leaves in scipy order; subtrees below `cut` are coloured by `colors[cluster]`."""
    n=Z.shape[0]+1; order=sl_dend(Z,no_plot=True)['leaves']; pos={leaf:k for k,leaf in enumerate(order)}
    hgt={}; 
    memb={i:[i] for i in range(n)}
    for k,(a,b,h,c) in enumerate(Z):
        a,b=int(a),int(b); pos[n+k]=(pos[a]+pos[b])/2; hgt[n+k]=h; memb[n+k]=memb[a]+memb[b]
    cl=None
    if cut is not None: cl=fcluster(Z,cut,'distance')-1
    for k,(a,b,h,c) in enumerate(Z):
        a,b=int(a),int(b); col=GREY
        if cl is not None and h<cut: col=colors[memb[n+k][0]]
        for ch in (a,b):
            y0=hgt.get(ch,0.0); ax.plot([pos[ch],pos[ch]],[y0,h],color=col,lw=lw,solid_capstyle='round')
        ax.plot([pos[a],pos[b]],[h,h],color=col,lw=lw,solid_capstyle='round')
    ax.set_xlim(-0.6,n-0.4); ax.set_ylim(0,ymax or Z[-1,2]*1.06)
    if show_leaf:
        ax.set_xticks(range(n)); ax.set_xticklabels([leaf_labels[l] for l in order],fontsize=fs)
    else: ax.set_xticks([])
    ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x',length=0)
    return order,pos
def draw_clusters(ax,X,labels,pad=0.55,fc_alpha=0.22,ms=34,colmap=None,names=None,fs=8.5,lw_edge=9):
    from scipy.spatial import ConvexHull
    for c in sorted(set(labels)):
        P=X[labels==c]; col=(colmap or CLU)[c%7]
        if len(P)>=3:
            h=ConvexHull(P); poly=P[h.vertices]
            ax.fill(poly[:,0],poly[:,1],color=col,alpha=fc_alpha,lw=0,zorder=1)
            ax.plot(*np.r_[poly,poly[:1]].T,color=col,alpha=0.22,lw=lw_edge,solid_joinstyle='round',zorder=1)
        elif len(P)==2:
            ax.plot(P[:,0],P[:,1],color=col,alpha=0.30,lw=lw_edge*1.8,solid_capstyle='round',zorder=1)
        else:
            ax.scatter(P[:,0],P[:,1],s=380,color=col,alpha=0.22,lw=0,zorder=1)
        ax.scatter(P[:,0],P[:,1],s=ms,color=col,marker=MK[c%7],edgecolor='white',lw=0.7,zorder=5)
def finch_axes(ax,xl=True,yl=True,fs=8.5):
    ax.set_xlim(-1.5,12.7); ax.set_ylim(0.2,13.8); ax.set_xticks([0,4,8,12]); ax.set_yticks([2,6,10]); ax.set_aspect('equal')
    ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    if xl: ax.set_xlabel('bill length (mm)',fontsize=9)
    if yl: ax.set_ylabel('wing length (cm)',fontsize=9)
def label_pts(ax,X,dx=0.35,dy=0.35,fs=8.5,names=NM,col=NAVY):
    for k,p in enumerate(X): ax.text(p[0]+dx-(1.15 if names[k]=='C' else 0),p[1]+dy,names[k],fontsize=fs,color=col,zorder=6,weight='bold')

# ---- 37.0-1 opener: the data and the tree
fig,axs=plt.subplots(1,2,figsize=(5.4,2.45),dpi=240,gridspec_kw={'width_ratios':[1.0,1.0]})
ax=axs[0]; draw_clusters(ax,X7,lab3); label_pts(ax,X7); finch_axes(ax)
ax=axs[1]; order,pos=dendro(ax,Zw,list(NM),cut=7.0,colors=[CLU[f] for f in fam]); ax.set_ylabel('merge height $h$',fontsize=9)
ax.axhline(7.0,color=PUR,lw=1.0,ls='--'); ax.text(-0.45,7.4,'cut: 3 families',fontsize=8.5,color=PUR,va='bottom')
ax.set_yticks([0,5,10,15])
fig.tight_layout(w_pad=1.0); sv(fig,'f370.png',470)
NUM['ward7']={'heights':[round(float(h),3) for h in Zw[:,2]],'order':[NM[i] for i in order]}
print('f370 ok')

# ---- 37.2-1 growing the tree: after 0, 2, 4, 6 merges (Ward)
stages=[(0,'7 clusters'),(2,'5 clusters'),(4,'3 clusters'),(6,'1 cluster')]
fig,axs=plt.subplots(1,4,figsize=(5.4,1.8),dpi=240,sharey=True)
n=7; memb={i:[i] for i in range(n)}
for k,(a,b,h,c) in enumerate(Zw): memb[n+k]=memb[int(a)]+memb[int(b)]
for ax,(m,title) in zip(axs,stages):
    alive=set(range(n))
    for k in range(m):
        a,b=int(Zw[k,0]),int(Zw[k,1]); alive-={a,b}; alive.add(n+k)
    lab=np.zeros(n,int); cols={}
    for c,node in enumerate(sorted(alive,key=lambda q:min(memb[q]))):
        lab[memb[node]]=c; fs_=set(fam[memb[node]]); cols[c]=CLU[fam[memb[node][0]]] if len(fs_)==1 else GREY
    from scipy.spatial import ConvexHull
    for c in sorted(set(lab)):
        P=X7[lab==c]; col=cols[c]; fm=fam[lab==c][0]; mk=MK[fm] if m<6 else 'o'
        if len(P)>=3:
            hh=ConvexHull(P); poly=P[hh.vertices]; ax.fill(poly[:,0],poly[:,1],color=col,alpha=0.22,lw=0,zorder=1); ax.plot(*np.r_[poly,poly[:1]].T,color=col,alpha=0.22,lw=7,solid_joinstyle='round',zorder=1)
        elif len(P)==2: ax.plot(P[:,0],P[:,1],color=col,alpha=0.30,lw=12,solid_capstyle='round',zorder=1)
        else: ax.scatter(P[:,0],P[:,1],s=230,color=col,alpha=0.22,lw=0,zorder=1)
        ax.scatter(P[:,0],P[:,1],s=22,color=col,marker=mk,edgecolor='white',lw=0.7,zorder=5)
    if m==0: label_pts(ax,X7,dx=0.5,dy=0.2,fs=8)
    ax.set_xlim(-1.5,12.7); ax.set_ylim(0.2,13.8); ax.set_xticks([0,6,12]); ax.set_yticks([2,6,10]); ax.set_aspect('equal'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title,fontsize=8.5,color=NAVY)
fig.tight_layout(w_pad=0.2); sv(fig,'f372.png',470); print('f372')

# ---- 37.3-1 four ways to measure the distance between clusters ABC and DE
A_=X7[:3]; B_=X7[3:5]
Dm=np.sqrt(((A_[:,None]-B_[None])**2).sum(-1)); NUM['abc_de']={'single':round(float(Dm.min()),3),'complete':round(float(Dm.max()),3),'average':round(float(Dm.mean()),3)}
muA,muB=A_.mean(0),B_.mean(0); dlt=3*2/5*float(((muA-muB)**2).sum()); NUM['abc_de']['ward_delta']=round(dlt,3); NUM['abc_de']['ward_height']=round(float(np.sqrt(2*dlt)),3); NUM['abc_de']['centroid_dist']=round(float(np.linalg.norm(muA-muB)),3)
fig,axs=plt.subplots(1,4,figsize=(5.4,1.95),dpi=240,sharey=True)
ttl=['single: smallest','complete: largest','average: mean of 6','Ward: centroids']
vals=[f"{NUM['abc_de']['single']:.2f}",f"{NUM['abc_de']['complete']:.2f}",f"{NUM['abc_de']['average']:.2f}",f"Δ = {dlt:.2f}"]
for k,ax in enumerate(axs):
    for P,col,mk in ((A_,BLUE,'o'),(B_,PUR,'^')):
        ax.scatter(P[:,0],P[:,1],s=24,color=col,marker=mk,edgecolor='white',lw=0.6,zorder=5)
    if k==0:
        i,j=np.unravel_index(Dm.argmin(),Dm.shape); ax.plot([A_[i,0],B_[j,0]],[A_[i,1],B_[j,1]],color=RED,lw=1.8,zorder=3)
    if k==1:
        i,j=np.unravel_index(Dm.argmax(),Dm.shape); ax.plot([A_[i,0],B_[j,0]],[A_[i,1],B_[j,1]],color=RED,lw=1.8,zorder=3)
    if k==2:
        for p in A_:
            for q in B_: ax.plot([p[0],q[0]],[p[1],q[1]],color=RED,lw=0.9,alpha=0.8,zorder=3)
    if k==3:
        ax.plot([muA[0],muB[0]],[muA[1],muB[1]],color=RED,lw=1.8,zorder=3)
        ax.scatter([muA[0]],[muA[1]],s=60,marker='+',color=GREEN,lw=1.8,zorder=6); ax.scatter([muB[0]],[muB[1]],s=60,marker='+',color=GREEN,lw=1.8,zorder=6)
    ax.set_title(ttl[k],fontsize=8.5,color=NAVY); ax.text(0.5,0.04,vals[k],transform=ax.transAxes,ha='center',fontsize=9,color=RED,weight='bold')
    ax.set_xlim(-1,12.2); ax.set_ylim(0.2,6.2); ax.set_xticks([0,6,12]); ax.set_yticks([2,4]); ax.set_aspect('equal'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    for kk,(dx_,dy_) in enumerate([(0.0,0.55),(0.45,0.5),(-0.6,0.5),(0.0,0.55),(0.0,0.55)]): ax.text(X7[kk,0]+dx_,X7[kk,1]+dy_,NM[kk],fontsize=8,color=GREY,ha='center',weight='bold',zorder=6)
fig.tight_layout(w_pad=0.2); sv(fig,'f373.png',470); print('f373')

# ---- 37.3-2 single linkage is the minimum spanning tree
D7=squareform(pdist(X7)); T=minimum_spanning_tree(D7).tocoo(); edges=sorted(zip(T.data,T.row,T.col)); NUM['mst']=[(round(float(w),3),NM[min(i,j)]+NM[max(i,j)]) for w,i,j in edges]
Zs=sl_link(X7,'single'); NUM['single_heights']=[round(float(h),3) for h in Zs[:,2]]
assert np.allclose(sorted(T.data),Zs[:,2])
fig,axs=plt.subplots(1,2,figsize=(5.4,2.4),dpi=240)
ax=axs[0]
LP={'BC':(0.5,2.4,'center','bottom'),'AB':(1.7,3.0,'center','bottom'),'AD':(5.0,3.75,'center','bottom'),'DE':(9.6,3.5,'left','bottom'),'FG':(6.9,10.9,'left','bottom'),'DG':(7.7,6.4,'left','center')}
for rank,(w,i,j) in enumerate(edges):
    cut=rank>=len(edges)-2
    ax.plot([X7[i,0],X7[j,0]],[X7[i,1],X7[j,1]],color=RED if cut else GREEN,lw=1.6,ls=(0,(3,2)) if cut else '-',zorder=3)
    key=NM[min(i,j)]+NM[max(i,j)]; lx,ly,ha,va=LP[key]
    ax.text(lx,ly,f'{w:.2f}',fontsize=8,color=RED if cut else GREEN,ha=ha,va=va,zorder=6)
lab_s=fcluster(Zs,3,'maxclust'); NUM['single_k3']=[int(v) for v in lab_s]
ax.scatter(X7[:,0],X7[:,1],s=26,color=BLUE,edgecolor='white',lw=0.7,zorder=5)
PL={'A':(0.35,0.3,'left','bottom'),'B':(0.1,-0.35,'center','top'),'C':(-0.35,0.0,'right','center'),'D':(0.1,-0.35,'left','top'),'E':(0.3,-0.35,'left','top'),'F':(-0.35,0.0,'right','center'),'G':(0.35,0.0,'left','center')}
for k,p in enumerate(X7):
    dx,dy,ha,va=PL[NM[k]]; ax.text(p[0]+dx,p[1]+dy,NM[k],fontsize=8.5,color=NAVY,weight='bold',ha=ha,va=va,zorder=6)
finch_axes(ax); ax.set_xlim(-2.5,12.7)
ax=axs[1]; order_s,pos_s=dendro(ax,Zs,list(NM),cut=4.4,colors=[CLU[0],CLU[1],CLU[2]][:1]*7 if False else [BLUE,BLUE,BLUE,BLUE,OR,PUR,PUR]); ax.set_ylabel('merge height $h$',fontsize=9); ax.set_yticks([0,2,4,6]); ax.axhline(4.4,color=PUR,lw=0.9,ls='--')
ax.text(6.35,4.55,'cut',fontsize=8.5,color=PUR,ha='right',va='bottom')
fig.tight_layout(w_pad=1.0); sv(fig,'f374.png',470); print('f374')

# ---- 37.4-1 structured Ward on a swiss roll
Xr,tr=make_swiss_roll(1500,noise=0.05,random_state=0); Xr2=Xr[:,[0,2]]
con=kneighbors_graph(Xr,n_neighbors=10,include_self=False)
fig,axs=plt.subplots(1,3,figsize=(5.4,2.0),dpi=240)
NUM['roll']={}
for ax,(nm,cn) in zip(axs[1:],(('without connectivity',None),('with a 10-neighbour graph',con))):
    t0=time.time(); m=AC(n_clusters=6,linkage='ward',connectivity=cn).fit(Xr); el=time.time()-t0
    # width of each cluster along the roll (range of the roll parameter t)
    rng_t=[float(tr[m.labels_==c].max()-tr[m.labels_==c].min()) for c in range(6)]
    NUM['roll'][nm]={'sizes':sorted(np.bincount(m.labels_).tolist()),'t_range_max':round(max(rng_t),2),'t_range_mean':round(float(np.mean(rng_t)),2),'seconds':round(el,3),'components':int(m.n_connected_components_)}
    cm=[BLUE,OR,PUR,TEAL,MAG,GREEN]
    for c in range(6): ax.scatter(Xr2[m.labels_==c,0],Xr2[m.labels_==c,1],s=1.8,color=cm[c],lw=0)
    ax.set_title(nm,fontsize=8.5,color=NAVY); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
ax=axs[0]; sc=ax.scatter(Xr2[:,0],Xr2[:,1],s=1.8,c=tr,cmap='viridis',lw=0); ax.set_title('the data (colour: place on the sheet)',fontsize=8.5,color=NAVY); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
fig.tight_layout(w_pad=0.4); sv(fig,'f375.png',470); print('f375',NUM['roll'])

# ---- 37.5-1 BIRCH: the threshold sets the number of subclusters
centres=np.array([[0,0],[5,0],[2.5,4.5]]); Xb,yb=make_blobs(n_samples=600,centers=centres,cluster_std=0.9,random_state=0)
fig,axs=plt.subplots(1,3,figsize=(5.4,2.15),dpi=240,sharey=True); NUM['birch']={}
for ax,T in zip(axs,(0.3,0.8,1.5)):
    b=Birch(threshold=T,n_clusters=3).fit(Xb); sc=b.subcluster_centers_; sl=b.subcluster_labels_
    a_=ari(yb,b.labels_); NUM['birch'][str(T)]={'subclusters':int(len(sc)),'ari':round(float(a_),3),'sizes':np.bincount(b.labels_).tolist()}
    ax.scatter(Xb[:,0],Xb[:,1],s=3,color=GRID,lw=0,zorder=1)
    for (x_,y_),l in zip(sc,sl):
        ax.add_patch(Circle((x_,y_),T,fc='none',ec=CLU[l%3],lw=0.7,alpha=0.9,zorder=3))
    for l in range(3): ax.scatter(sc[sl==l,0],sc[sl==l,1],s=7,color=CLU[l],marker=MK[l],lw=0,zorder=4)
    ax.set_title(f'$T$ = {T}: {len(sc)} subclusters\nARI {a_:.3f}',fontsize=8.2,color=NAVY); ax.set_aspect('equal'); ax.set_xticks([-2,2,6]); ax.set_yticks([-2,2,6]); ax.set_xlim(-3.2,8.2); ax.set_ylim(-3.8,7.8)
fig.tight_layout(w_pad=0.3); sv(fig,'f376.png',470); print('f376',NUM['birch'])

# ---- 37.7-1 flow chart of fit
fig,ax=new(5.4,1.9)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.4): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(0.75,1.0,1.3,1.5,'Input\n$\\mathbf{X}$, linkage,\nmetric,\nconnectivity',fc=LGREY,ec=GREY)
fb(2.45,1.0,1.4,1.5,'All pairwise\ndistances\n(or only the\ngraph edges)')
fb(4.35,1.0,1.9,1.5,'Closest pair $(A, B)$:\nrecord children_,\ndistances_ $= h$;\nmerge, update the\ndistances (Eq. 37.3-6)',fc='#FDF0E0',ec=OR)
fb(6.2,1.0,1.3,1.5,'Cut the tree:\n$K$ clusters or\nheight $t$\n→ labels_',fc='#E3F4EA',ec=GREEN)
A((1.4,1.0),(1.75,1.0)); A((3.15,1.0),(3.4,1.0)); A((5.3,1.0),(5.55,1.0))
ax.add_patch(FancyArrowPatch((3.9,1.78),(4.8,1.78),connectionstyle='arc3,rad=-1.6',arrowstyle='-|>',mutation_scale=8,color=NAVY,lw=0.9))
ax.text(4.35,2.55,'repeat $n-1$ times',fontsize=8.4,color=GREY,ha='center',style='italic')
ax.set_xlim(0,6.95); ax.set_ylim(0.1,2.75); sv(fig,'f377.png',400); print('f377')

# ---- 37.10-1 gallery: four linkages on three data sets
rng=np.random.default_rng(3)
Xm,ym=make_moons(150,noise=0.06,random_state=0); Xc,yc=make_circles(150,factor=0.4,noise=0.05,random_state=0)
Xk,yk=make_blobs(n_samples=135,centers=[[0,0],[4,0],[2,3.4]],cluster_std=0.55,random_state=1)
Xk=np.r_[Xk,rng.uniform([-2,-2],[6,5.5],size=(15,2))]; yk=np.r_[yk,[3]*15]       # label 3 = scattered noise points
SETS=[('two moons',Xm,ym,2),('two rings',Xc,yc,2),('blobs and noise',Xk,yk,3)]
LINKS=['single','complete','average','ward']
fig,axs=plt.subplots(3,4,figsize=(5.4,3.35),dpi=240); NUM['gallery']={}
for r,(nm,X_,y_,K) in enumerate(SETS):
    for c,L in enumerate(LINKS):
        ax=axs[r,c]; m=AC(n_clusters=K,linkage=L).fit(X_); a_=ari(y_,m.labels_); NUM['gallery'][f'{nm}/{L}']={'ari':round(float(a_),3),'sizes':sorted(np.bincount(m.labels_).tolist(),reverse=True)}
        for k in range(K): ax.scatter(X_[m.labels_==k,0],X_[m.labels_==k,1],s=6,color=CLU[k],marker=MK[k],lw=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal'); ax.spines['left'].set_visible(True); ax.spines['bottom'].set_visible(True)
        if r==0: ax.set_title(L,fontsize=9,color=NAVY)
        ax.set_xlabel(f'ARI {a_:.2f}',fontsize=8.5,labelpad=1.5)
        if c==0: ax.set_ylabel(nm,fontsize=8.8)
fig.tight_layout(w_pad=0.3,h_pad=0.3); sv(fig,'f3710a.png',470); print('f3710a')

# ---- 37.10-2 the knob K (ARI against K) and the cost (time against n)
Xq,yq=make_blobs(n_samples=400,centers=[[0,0],[4,0],[2,3.5],[6.5,3.5]],cluster_std=[0.8,0.8,0.8,0.8],random_state=2)
KS=list(range(2,9)); NUM['ari_vs_k']={}
fig,axs=plt.subplots(1,2,figsize=(5.4,2.3),dpi=240)
ax=axs[0]
for L,col,mk,ls in zip(LINKS,[RED,OR,GREY,GREEN],['o','s','^','D'],['-','--','-.','-']):
    v=[ari(yq,AC(n_clusters=k,linkage=L).fit_predict(Xq)) for k in KS]; NUM['ari_vs_k'][L]=[round(float(a),3) for a in v]
    ax.plot(KS,v,ls=ls,marker=mk,ms=3.6,lw=1.3,color=col,label=L)
ax.axvline(4,color=PUR,lw=1.0,ls='--'); ax.set_xlabel('number of clusters $K$'); ax.set_ylabel('ARI'); ax.set_ylim(-0.02,1.05); ax.legend(loc='center right',bbox_to_anchor=(1.0,0.38),fontsize=8.2,borderaxespad=0.2,handlelength=1.8); grid(ax)
T=json.load(open('verify/timing.json')); ax=axs[1]; ns=np.array(T['n'])
for L,col,mk,ls in zip(['ward','average','single','birch'],[GREEN,GREY,RED,PUR],['D','^','o','s'],['-','-.','-','--']):
    ax.plot(ns,T[L],ls=ls,marker=mk,ms=3.6,lw=1.3,color=col,label=L if L!='birch' else 'Birch')
ref=np.array(T['ward'])[2]*(ns/ns[2])**2; ax.plot(ns,ref,color=GREY,lw=0.8,ls=':'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([500,2000,8000]); ax.set_xticklabels(['500','2000','8000']); ax.minorticks_off(); ax.set_yticks([0.01,0.1,1]); ax.set_yticklabels(['0.01','0.1','1']); ax.set_xlabel('samples $n$'); ax.set_ylabel('fit time (s)'); ax.legend(loc='upper left',fontsize=8.2,borderaxespad=0.2,handlelength=1.8); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f3710b.png',470); print('f3710b',NUM['ari_vs_k'])

# ---- 37.12-1 wine: the Ward tree with the cultivars underneath, and the ARI of each variant
Xw,yw=load_wine(return_X_y=True); Xws=StandardScaler().fit_transform(Xw); Zwine=sl_link(Xws,'ward')
mw=AC(n_clusters=3).fit(Xws); chk=fcluster(Zwine,3,'maxclust'); assert len(set(zip(chk,mw.labels_)))==3
NUM['wine']={'top_heights':[round(float(h),2) for h in Zwine[-6:,2]]}
fig=plt.figure(figsize=(5.4,2.2),dpi=240); gs=fig.add_gridspec(2,2,height_ratios=[1,0.07],width_ratios=[1.25,1],hspace=0.02,wspace=0.28)
ax=fig.add_subplot(gs[0,0]); rug=fig.add_subplot(gs[1,0],sharex=ax)
order_w=sl_dend(Zwine,no_plot=True)['leaves']; cut_h=(Zwine[-3,2]+Zwine[-2,2])/2
# colour each of the three branches by the cluster id (sorted by first leaf in plotting order)
cl3=fcluster(Zwine,3,'maxclust'); ids=[]; [ids.append(cl3[l]) for l in order_w if cl3[l] not in ids]; colw={c:CLU[k] for k,c in enumerate(ids)}
leafcol=[NAVY]*len(Xws)
dendro(ax,Zwine,None,cut=cut_h,colors=leafcol,lw=0.8,show_leaf=False,ymax=Zwine[-1,2]*1.04)
ax.axhline(cut_h,color=PUR,lw=0.9,ls='--'); ax.set_ylabel('merge height $h$',fontsize=9); ax.set_xticks([]); ax.set_yticks([0,10,20,30])
ax.text(2,cut_h+0.8,'cut: 3 clusters',fontsize=8.3,color=PUR,va='bottom')
for k,l in enumerate(order_w): rug.add_patch(Rectangle((k-0.5,0),1,1,fc=[BLUE,OR,GREEN][yw[l]],ec='none'))
rug.set_xlim(-0.6,len(order_w)-0.4); rug.set_ylim(0,1); rug.axis('off')
rug.text(-0.012,0.5,'cultivar',transform=rug.transAxes,fontsize=8.3,color=GREY,ha='right',va='center')
ax=fig.add_subplot(gs[:,1])
names=['single','complete','average','Ward','k-means']; raw=[]; sc=[]
for L in LINKS: raw.append(ari(yw,AC(n_clusters=3,linkage=L).fit_predict(Xw))); sc.append(ari(yw,AC(n_clusters=3,linkage=L).fit_predict(Xws)))
raw.append(ari(yw,KMeans(3,n_init=10,random_state=0).fit_predict(Xw))); sc.append(ari(yw,KMeans(3,n_init=10,random_state=0).fit_predict(Xws)))
NUM['wine']['ari_raw']=[round(float(a),3) for a in raw]; NUM['wine']['ari_scaled']=[round(float(a),3) for a in sc]
yy=np.arange(5)[::-1]; ax.barh(yy+0.19,raw,0.36,color=GREY,hatch='////',edgecolor='white',label='raw features'); ax.barh(yy-0.19,sc,0.36,color=BLUE,label='standardised')
for y_,a,b in zip(yy,raw,sc): ax.text(max(a,0)+0.015,y_+0.19,f'{a:.2f}',fontsize=8,va='center',color=NAVY); ax.text(max(b,0)+0.015,y_-0.19,f'{b:.2f}',fontsize=8,va='center',color=NAVY)
ax.set_yticks(yy); ax.set_yticklabels(names); ax.set_xlim(-0.02,1.15); ax.set_xlabel('ARI against the cultivars'); ax.legend(loc='upper right',bbox_to_anchor=(1.0,0.97),fontsize=8.2,borderaxespad=0.2,handlelength=1.4); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
sv(fig,'f3712a.png',470); print('f3712a',NUM['wine'])

# ---- 37.12-2 digits: FeatureAgglomeration of the 64 pixels
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
Xd,yd=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.25,random_state=0,stratify=yd)
gridg=grid_to_graph(8,8)
def acc_(Ztr,Zte): return make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Ztr,ytr).score(Zte,yte)
KSd=[2,4,8,16,24,32,48]; accf=[];accg=[]
for k in KSd:
    f=FA(n_clusters=k).fit(Xtr); g=FA(n_clusters=k,connectivity=gridg).fit(Xtr)
    accf.append(acc_(f.transform(Xtr),f.transform(Xte))); accg.append(acc_(g.transform(Xtr),g.transform(Xte)))
raw_acc=acc_(Xtr,Xte); NUM['digits']={'k':KSd,'free':[round(float(a),4) for a in accf],'grid':[round(float(a),4) for a in accg],'raw':round(float(raw_acc),4)}
f16=FA(n_clusters=16).fit(Xtr); g16=FA(n_clusters=16,connectivity=gridg).fit(Xtr)
pal=ListedColormap(plt.cm.tab20(np.arange(16)/20.0+0.0)) if False else ListedColormap([plt.cm.tab20(i) for i in range(16)])
fig,axs=plt.subplots(1,4,figsize=(5.4,1.5),dpi=240,gridspec_kw={'width_ratios':[1,1,2.05,2.2]})
for ax,fm,ttl in ((axs[0],f16,'16 groups, free'),(axs[1],g16,'16 groups, grid')):
    ax.imshow(fm.labels_.reshape(8,8),cmap=pal,vmin=0,vmax=15,interpolation='nearest')
    for i in range(9): ax.axhline(i-0.5,color='white',lw=0.5); ax.axvline(i-0.5,color='white',lw=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=8.3,color=NAVY)
    for s_ in ax.spines.values(): s_.set_visible(False)
dg=Xte[np.flatnonzero(yte==3)[0]]; rf=f16.inverse_transform(f16.transform(dg[None]))[0]; rg=g16.inverse_transform(g16.transform(dg[None]))[0]
sep=np.full((8,1),np.nan); big=np.hstack([dg.reshape(8,8),sep,rf.reshape(8,8)])
ax=axs[2]; ax.imshow(np.ma.masked_invalid(big),cmap='Greys',vmin=0,vmax=16,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title('a digit, and from 16 free groups',fontsize=8.3,color=NAVY)
for s_ in ax.spines.values(): s_.set_visible(False)
ax=axs[3]; ax.plot(KSd,accf,'-o',ms=3.4,lw=1.3,color=GREEN,label='free'); ax.plot(KSd,accg,'--s',ms=3.4,lw=1.3,color=OR,label='grid'); ax.axhline(raw_acc,color=GREY,lw=0.9,ls=':'); ax.text(KSd[0],raw_acc+0.012,'all 64 pixels',fontsize=8.2,color=GREY,va='bottom')
ax.set_xlabel('groups $k$',fontsize=9); ax.set_ylabel('test accuracy',fontsize=9); ax.set_ylim(0.2,1.12); ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_xticks([2,16,32,48]); ax.legend(loc='lower right',fontsize=8.2,borderaxespad=0.2,handlelength=1.6); grid(ax)
fig.tight_layout(w_pad=0.5)
for a_ in axs[:3]: a_.set_anchor('C')
sv(fig,'f3712b.png',470); print('f3712b',NUM['digits'])
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)

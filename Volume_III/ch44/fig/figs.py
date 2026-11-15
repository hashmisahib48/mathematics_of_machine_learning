# Figures of Chapter 44 (no digits). Run: python3 fig/figs.py  (PNGs into ch44/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.decomposition import PCA
from sklearn.manifold import (Isomap, LocallyLinearEmbedding, SpectralEmbedding, MDS, TSNE, trustworthiness)
from sklearn.neighbors import kneighbors_graph
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
CM='viridis'

# ---------- data used by several figures
X,t=make_swiss_roll(1000,noise=0.0,random_state=0)

# ---- 44.0-1 opener: roll, PCA, Isomap
fig=plt.figure(figsize=(5.4,2.0),dpi=240)
a3=fig.add_axes([0.0,0.0,0.32,1.0],projection='3d'); a3.scatter(X[:,0],X[:,1],X[:,2],c=t,cmap=CM,s=3.5,lw=0)
a3.view_init(elev=12,azim=-72); a3.set_axis_off(); a3.set_box_aspect((1,0.85,1))
a3.set_title('swiss roll in 3-D',fontsize=9,color=NAVY,y=0.92)
Zp=PCA(2).fit_transform(X); Zi=Isomap(n_neighbors=10).fit_transform(X)
for k,(Z,ttl) in enumerate(((Zp,'PCA: layers overlap'),(Zi,'Isomap: unrolled'))):
    ax=fig.add_axes([0.37+0.33*k,0.08,0.29,0.78]); ax.scatter(Z[:,0],Z[:,1],c=t,cmap=CM,s=3.5,lw=0); ax.set_aspect('equal'); clean(ax); ax.set_title(ttl,fontsize=9,color=NAVY)
NUM['opener']={'trust_pca':float(trustworthiness(X,Zp,n_neighbors=10)),'trust_iso':float(trustworthiness(X,Zi,n_neighbors=10))}
sv(fig,'f440.png',470)

# ---- 44.2-1 spiral: chord, graph path, short circuit
th=np.linspace(1.5*np.pi,4.5*np.pi,110); S=np.c_[0.3*th*np.cos(th),0.3*th*np.sin(th)]
i0,i1=18,96   # two points on neighbouring whorls
def gdist(m):
    A=kneighbors_graph(S,m,mode='distance'); Dg,pred=shortest_path(A,method='D',directed=False,return_predecessors=True)
    return A,Dg,pred
def path(pred,i,j):
    p=[j]
    while p[-1]!=i: p.append(pred[i,p[-1]])
    return p[::-1]
fig,axs=plt.subplots(1,3,figsize=(5.4,1.95),dpi=240); NUM['spiral']={}
for ax,(m,ttl) in zip(axs,((None,'(a) chord and path'),(3,'(b) m = 3'),(14,'(c) m = 14, short circuit'))):
    ax.plot(*S.T,color=GRID,lw=1.0,zorder=1); ax.scatter(*S.T,s=5,color=BLUE,lw=0,zorder=3)
    if m is None:
        A,Dg,pred=gdist(3); ch=np.linalg.norm(S[i0]-S[i1]); NUM['spiral']['chord']=float(ch); NUM['spiral']['geodesic3']=float(Dg[i0,i1])
        ax.plot(*S[[i0,i1]].T,color=RED,lw=1.4,ls='--',zorder=4); pp=path(pred,i0,i1); ax.plot(*S[pp].T,color=GREEN,lw=1.6,zorder=4)
    else:
        A,Dg,pred=gdist(m); C=A.tocoo()
        for a,b in zip(C.row,C.col): ax.plot(*S[[a,b]].T,color=OR,lw=0.5,zorder=2)
        pp=path(pred,i0,i1); ax.plot(*S[pp].T,color=GREEN,lw=1.6,zorder=4)
        NUM['spiral'][f'geodesic{m}']=float(Dg[i0,i1]); NUM['spiral'][f'edges{m}']=int(A.nnz); NUM['spiral'][f'components{m}']=int(connected_components(A)[0])
    ax.scatter(*S[[i0,i1]].T,s=22,color=NAVY,zorder=6); ax.set_aspect('equal'); clean(ax); ax.set_title(ttl,fontsize=9,color=NAVY)
axs[0].text(S[i0,0]-0.15,S[i0,1]-0.5,'P',fontsize=9,color=NAVY,ha='right'); axs[0].text(S[i1,0]+0.15,S[i1,1]+0.3,'Q',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=0.4); sv(fig,'f441.png',470)

# ---- 44.4-1 MDS: villages, stress by iteration, Shepard diagram
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_digits
fig,axs=plt.subplots(1,3,figsize=(5.4,2.0),dpi=240,gridspec_kw={'width_ratios':[1.45,1,1]})
ax=axs[0]; V=np.array([[0,0],[4,0],[4,3],[0,3]],float); R=np.array([[-2.6458,-2.2913],[2.6458,-2.2913],[2.6458,2.2913],[-2.6458,2.2913]])+np.array([9.2,1.5])
ax.add_patch(plt.Polygon(V,fc='none',ec=BLUE,lw=1.2,zorder=2)); ax.add_patch(plt.Polygon(R,fc='none',ec=OR,lw=1.2,ls='--',zorder=2))
ax.scatter(*V.T,s=24,color=BLUE,zorder=5); ax.scatter(*R.T,s=24,color=OR,marker='s',zorder=5)
for P_,sg in ((V,1),(R,1)):
    for nm,p,(dx,dy) in zip('ABCD',P_,((-0.6,-0.45),(0.6,-0.45),(0.6,0.45),(-0.6,0.45))): ax.text(p[0]+dx,p[1]+dy,nm,fontsize=9,color=NAVY,ha='center',va='center')
ax.text(2,-0.95,'distances\n3, 4, 5',fontsize=8.5,color=BLUE,ha='center',va='top',linespacing=1.1); ax.text(9.2,-0.95,'side roads\nonly',fontsize=8.5,color=OR,ha='center',va='top',linespacing=1.1)
ax.set_aspect('equal'); ax.set_xlim(-1.3,12.5); ax.set_ylim(-2.3,4.5); clean(ax); ax.set_title('(a) two maps of four villages',fontsize=9,color=NAVY)
# stress per iteration
Xd=load_digits().data[:150]/16; Dm=squareform(pdist(Xd)); n=len(Dm); Z=np.random.RandomState(0).uniform(size=(n,2)); st=[]
for it in range(219):
    E=squareform(pdist(Z)); E[E==0]=1e-5; Rr=Dm/E; B=-Rr; B[np.diag_indices(n)]+=Rr.sum(1); Z=B@Z/n
    E=squareform(pdist(Z)); st.append(((E-Dm)**2).sum()/2)
st=np.array(st); NUM['smacof']={'stress':[float(s) for s in st[[0,1,4,9,29,99,218]]],'monotone':bool(np.all(np.diff(st)<=1e-12))}
ax=axs[1]; ax.plot(np.arange(1,220),st,color=RED,lw=1.5); ax.set_xlabel('iteration'); ax.set_ylabel('stress'); grid(ax); ax.set_title('(b) stress, Eq. 44.4-1',fontsize=9,color=NAVY); ax.set_xlim(0,220)
ax=axs[2]; ZP=PCA(2).fit_transform(Xd); dp=pdist(Xd); ax.scatter(dp,pdist(ZP),s=1.0,color=BLUE,lw=0,alpha=0.35,label='PCA'); ax.scatter(dp,pdist(Z),s=1.0,color=OR,lw=0,alpha=0.35,label='SMACOF')
mx=dp.max(); ax.plot([0,mx],[0,mx],color=GREEN,lw=1.0,ls='--'); ax.set_xlabel('original distance'); ax.set_ylabel('map distance'); grid(ax); ax.set_title('(c) Shepard diagram',fontsize=9,color=NAVY)
ax.set_xlim(0,mx); ax.set_ylim(0,mx); ax.set_aspect('equal'); ax.legend(loc='upper left',fontsize=8.5,markerscale=7,handletextpad=0.1,borderaxespad=0.1)
fig.tight_layout(w_pad=0.5); sv(fig,'f442.png',440)

# ---- 44.5-1 LLE weights and a Laplacian coordinate
fig,axs=plt.subplots(1,2,figsize=(5.0,1.9),dpi=240,gridspec_kw={'width_ratios':[1,1]})
ax=axs[0]; xi=np.array([1,1.]); nb=np.array([[0,0],[3,0.]]); w=np.array([2/3,1/3]); rec=w@nb
ax.plot(*nb.T,color=GRID,lw=1.0,zorder=1)
ax.scatter(*nb.T,s=34,color=BLUE,zorder=5); ax.scatter(*xi,s=40,color=NAVY,marker='*',zorder=6); ax.scatter(*rec,s=30,color=GREEN,marker='D',zorder=6)
ax.plot([xi[0],rec[0]],[xi[1],rec[1]],color=RED,lw=1.5,zorder=4)
ax.text(-0.15,-0.28,'$\\mathbf{x}_1$, weight 2/3',fontsize=8.5,color=NAVY,ha='left',va='top'); ax.text(3.15,-0.28,'$\\mathbf{x}_2$, 1/3',fontsize=8.5,color=NAVY,ha='right',va='top')
ax.text(1.12,1.12,'$\\mathbf{x}_i$',fontsize=9,color=NAVY,va='bottom'); ax.text(1.12,0.22,'$\\hat{\\mathbf{x}}_i$',fontsize=9,color=GREEN,va='bottom'); ax.text(1.75,0.55,'error 1',fontsize=8.5,color=RED,ha='left')
ax.set_xlim(-0.6,3.6); ax.set_ylim(-0.9,1.9); ax.set_aspect('equal'); clean(ax); ax.set_title('(a) one local reconstruction',fontsize=9,color=NAVY)
ax=axs[1]; th2=np.linspace(1.5*np.pi,4.5*np.pi,110); Sp=np.c_[0.3*th2*np.cos(th2),0.3*th2*np.sin(th2)]
z1=SpectralEmbedding(n_components=1,n_neighbors=4,random_state=0).fit_transform(Sp)[:,0]
sa=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Sp,axis=0),axis=1))]; NUM['lap1d']={'spearman':float(abs(spearmanr(z1,sa)[0]))}
sc=ax.scatter(sa,z1,c=sa,cmap=CM,s=8,lw=0); ax.set_xlabel('arc length along the spiral'); ax.set_ylabel('Laplacian coordinate'); grid(ax); ax.set_title('(b) a curve unrolled',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=0.6); sv(fig,'f443.png',440)

# ---- 44.7-1 t-SNE: kernels and forces
fig,axs=plt.subplots(1,2,figsize=(5.2,2.0),dpi=240)
d=np.linspace(0,6,300); ax=axs[0]
ax.plot(d,np.exp(-d**2/2),color=BLUE,lw=1.6,label='Gaussian, $\\sigma$ = 1'); ax.plot(d,1/(1+d**2),color=RED,lw=1.6,ls='--',label='Student-$t$, 1 d.f.')
ax.set_yscale('log'); ax.set_ylim(1e-6,1.5); ax.set_xlabel('distance $d$'); ax.set_ylabel('unnormalised weight'); grid(ax); ax.legend(loc='lower left',fontsize=8.5,handlelength=1.8,borderaxespad=0.2); ax.set_title('(a) tails',fontsize=9,color=NAVY)
ax=axs[1]; p_,Zn=0.004,200.0; w=1/(1+d**2); attr=4*p_*w*d; rep=4*(w/Zn)*w*d
ax.plot(d,attr,color=GREEN,lw=1.6,label='attraction $p_{ij}$'); ax.plot(d,rep,color=RED,lw=1.6,ls='--',label='repulsion $q_{ij}$'); ax.plot(d,attr-rep,color=NAVY,lw=1.0,ls=':',label='net')
dstar=float(np.sqrt(p_*Zn*0+1/(p_*Zn)-1)); ax.axvline(dstar,color=PUR,lw=1.0,ls='--'); ax.axhline(0,color=GREY,lw=0.5)
ax.set_xlabel('distance $d$ in the map'); ax.set_ylabel('force on $\\mathbf{z}_i$'); ax.set_xlim(0,6); ax.set_ylim(-0.003,0.011); grid(ax); ax.legend(loc='upper right',fontsize=8.5,handlelength=1.8,borderaxespad=0.2); ax.set_title('(b) forces for one pair',fontsize=9,color=NAVY)
ax.text(dstar+0.15,-0.0022,'$q_{ij}=p_{ij}$',fontsize=8.5,color=PUR,ha='left'); NUM['tsne_force']={'p':p_,'Z':Zn,'dstar':dstar}
fig.tight_layout(w_pad=0.7); sv(fig,'f444.png',450)

# ---- 44.8-1 flow chart
fig,ax=plt.subplots(figsize=(5.4,2.75),dpi=240); ax.set_xlim(0,10.8); ax.set_ylim(0,6.0); ax.axis('off')
def bx(x,y,w,h,t,fc,ec,fs=8.6,bold=False): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=fs,bold=bold,r=0.1)
cols=[(0.05,2.5),(2.8,2.3),(5.35,3.3),(8.9,1.85)]
heads=['input','matrix or objective','solved by','method']
for (x,w),h in zip(cols,heads): ax.text(x+w/2,5.75,h,ha='center',va='center',fontsize=9,color=NAVY,weight='bold')
rows=[('Classical MDS','all distances $d_{ij}$',r'$\mathbf{B}=-\frac{1}{2}\mathbf{JD}^{2}\mathbf{J}$','top eigenvectors, $\\sqrt{\\lambda}$ scaling'),
      ('Isomap','$m$-NN graph, geodesics',r'$\mathbf{B}$ from geodesics','top eigenvectors, $\\sqrt{\\lambda}$ scaling'),
      ('SMACOF MDS','all distances $d_{ij}$','stress of Eq. 44.4-1','Guttman transform, repeated'),
      ('LLE','$m$-NN, weights $w_{ij}$',r'$\mathbf{M}=(\mathbf{I}-\mathbf{W})^{T}(\mathbf{I}-\mathbf{W})$','bottom eigenvectors, skip first'),
      ('Laplacian','$m$-NN affinities $W_{ij}$',r'$\mathbf{L}=\mathbf{D}-\mathbf{W}$','bottom eigenvectors, $\\mathbf{L}z=\\lambda\\mathbf{D}z$'),
      ('t-SNE','perplexity, $p_{ij}$','KL$(P\\Vert Q)$ of Eq. 44.7-5','gradient descent, momentum')]
for k,(nm,a,b,c_) in enumerate(rows):
    y=4.75-k*0.88
    fc='#E3F4EA' if k<2 or k in (3,4) else ('#FDF0E0')
    bx(cols[0][0],y,cols[0][1],0.66,a,'white',BLUE,fs=8.6); bx(cols[1][0],y,cols[1][1],0.66,b,'white',BLUE,fs=8.6)
    bx(cols[2][0],y,cols[2][1],0.66,c_,'#E3F4EA' if k not in (2,5) else '#FDF0E0',GREEN if k not in (2,5) else OR,fs=8.6)
    bx(cols[3][0],y,cols[3][1],0.66,nm,'#E3EEFB',NAVY,fs=8.6,bold=True)
    for (x0,w0),(x1,w1) in zip(cols[:-1],cols[1:]): arr(ax,(x0+w0+0.02,y+0.33),(x1-0.02,y+0.33),c=GREY,lw=1.0,ms=6)
fig.subplots_adjust(0,0,1,1); sv(fig,'f445.png',420)

# ---- 44.12-1 Isomap as the neighbourhood grows (noisy roll)
Xn,tn=make_swiss_roll(1000,noise=0.5,random_state=0)
fig,axs=plt.subplots(1,4,figsize=(5.4,1.45),dpi=240); NUM['isogallery']={}
for ax,m in zip(axs,(5,10,20,40)):
    Z=Isomap(n_neighbors=m).fit_transform(Xn); T=trustworthiness(Xn,Z,n_neighbors=10); NUM['isogallery'][m]=float(T)
    ax.scatter(Z[:,0],Z[:,1],c=tn,cmap=CM,s=2.5,lw=0); ax.set_aspect('equal',adjustable='datalim'); clean(ax); ax.set_title(f'm = {m}',fontsize=9,color=PUR,pad=2); ax.text(0.5,-0.04,f'T(10) = {T:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color=NAVY)
fig.subplots_adjust(left=0.01,right=0.99,top=0.88,bottom=0.12,wspace=0.06); sv(fig,'f446.png',440)

# ---- 44.12-2 validation curves: trustworthiness against the neighbourhood size
ms=[4,5,6,8,10,12,15,20,30,40]; curves={'Isomap':[],'LLE':[],'Laplacian eigenmaps':[]}
for m in ms:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        curves['Isomap'].append(trustworthiness(Xn,Isomap(n_neighbors=m).fit_transform(Xn),n_neighbors=10))
        curves['LLE'].append(trustworthiness(Xn,LocallyLinearEmbedding(n_neighbors=m,random_state=0).fit_transform(Xn),n_neighbors=10))
        curves['Laplacian eigenmaps'].append(trustworthiness(Xn,SpectralEmbedding(n_neighbors=m,random_state=0).fit_transform(Xn),n_neighbors=10))
NUM['valcurve_roll']={'m':ms,**{k:[round(float(v),4) for v in vv] for k,vv in curves.items()}}
sc_=json.load(open('verify/digits_scan.json'))
fig,axs=plt.subplots(1,2,figsize=(5.2,2.1),dpi=240)
sty={'Isomap':(BLUE,'-','o'),'LLE':(GREEN,'--','s'),'Laplacian eigenmaps':(OR,':','^')}
ax=axs[0]
for k,v in curves.items(): c_,ls,mk=sty[k]; ax.plot(ms,v,ls,marker=mk,color=c_,ms=3.5,lw=1.4,label=k)
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks([5,10,20,40]); ax.set_xticklabels(['5','10','20','40']); ax.set_xlabel('neighbours $m$'); ax.set_ylabel('trustworthiness T(10)'); ax.set_ylim(0.75,1.01); grid(ax); ax.set_title('(a) noisy swiss roll',fontsize=9,color=NAVY)
ax.legend(loc='lower left',fontsize=8.0,handlelength=1.8,borderaxespad=0.2,bbox_to_anchor=(0.0,0.0))
ax.set_ylim(0.75,1.03)
ax=axs[1]; dm=[int(k) for k in sc_['Isomap']]; ax.plot(dm,[sc_['Isomap'][str(k)][1] for k in dm],'-o',color=BLUE,ms=3.5,lw=1.4,label='Isomap')
lm=sorted({int(k.split(',')[0]) for k in sc_['LLE']}); ax.plot(lm,[sc_['LLE'][f'{k},0.1'] for k in lm],'--s',color=GREEN,ms=3.5,lw=1.4,label='LLE, reg 0.1'); ax.plot(lm,[sc_['LLE'][f'{k},0.001'] for k in lm],'-.s',color=PUR,ms=3.5,lw=1.2,label='LLE, reg 0.001')
sm_=[int(k) for k in sc_['Laplacian']]; ax.plot(sm_,[sc_['Laplacian'][str(k)] for k in sm_],':^',color=OR,ms=3.5,lw=1.4,label='Laplacian')
ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks([6,10,20,30]); ax.set_xticklabels(['6','10','20','30']); ax.set_xlabel('neighbours $m$'); ax.set_ylim(0.70,0.98); grid(ax); ax.set_title('(b) digits (training images)',fontsize=9,color=NAVY)
ax.legend(loc='lower left',fontsize=8.0,handlelength=1.8,borderaxespad=0.2,labelspacing=0.25,ncol=2,columnspacing=0.8)
fig.tight_layout(w_pad=0.8); sv(fig,'f447.png',450)

# ---- 44.14-1 swiss roll and S-curve gallery
E=np.load('verify/swiss_embeddings.npz'); order=['PCA','MDS','Isomap','LLE','Hessian LLE','Laplacian','t-SNE']
names={'PCA':'PCA','MDS':'SMACOF MDS','Isomap':'Isomap','LLE':'LLE','Hessian LLE':'Hessian LLE','Laplacian':'Laplacian eigenmaps','t-SNE':'t-SNE'}
fig=plt.figure(figsize=(4.6,2.55),dpi=240); NUM['gallery']={}
for dn,mk in (('swiss roll',make_swiss_roll),('S-curve',make_s_curve)):
    Xg,tg=mk(1000,random_state=0)
    for key in order: NUM['gallery'][f'{dn}|{key}']=float(trustworthiness(Xg,E[f'{dn}|{key}'],n_neighbors=10))
Xg,tg=make_swiss_roll(1000,random_state=0); dn='swiss roll'
for j,key in enumerate(['data']+order):
    row=j//4; col=j%4; pos=[col*0.25+0.01,0.99-(row+1)*0.5+0.06,0.23,0.30]
    if key=='data':
        ax=fig.add_axes([col*0.25-0.01,0.99-(row+1)*0.5+0.02,0.26,0.38],projection='3d'); ax.scatter(Xg[:,0],Xg[:,1],Xg[:,2],c=tg,cmap=CM,s=2.5,lw=0)
        ax.view_init(elev=12,azim=-72); ax.set_axis_off(); ax.set_box_aspect((1,0.85,1)); ax.text2D(0.5,0.98,'swiss roll',transform=ax.transAxes,ha='center',va='top',fontsize=9,color=NAVY,weight='bold')
        continue
    Z=E[f'{dn}|{key}']; T=NUM['gallery'][f'{dn}|{key}']
    ax=fig.add_axes(pos); ax.scatter(Z[:,0],Z[:,1],c=tg,cmap=CM,s=2.2,lw=0); ax.set_aspect('equal',adjustable='datalim'); clean(ax)
    ax.set_title(names[key],fontsize=8.8,color=NAVY,pad=1); ax.text(0.5,-0.01,f'T(10) = {T:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color=GREY)
sv(fig,'f4410.png',440)
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print('figs ok')

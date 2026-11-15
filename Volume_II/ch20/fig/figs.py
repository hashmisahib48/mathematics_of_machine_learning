# All figures of Chapter 20. Run: python3 fig/figs.py [names]  (writes PNGs into ch20/ and numbers into verify/fig_numbers.json)
import sys,os,json,heapq; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, KDTree, NeighborhoodComponentsAnalysis, NearestCentroid
from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
ONLY=set(sys.argv[1:]); NF='verify/fig_numbers.json'
NUM=json.load(open(NF)) if os.path.exists(NF) else {}
def want(n): return (not ONLY) or n in ONLY
def fontchk(fig,name,printed_in):
    """smallest printed text size in points (design size x printed width / figure width)"""
    fw=fig.get_size_inches()[0]; sm=99; who=''
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            fs=t.get_fontsize()*printed_in/fw
            if fs<sm: sm=fs; who=t.get_text()[:25]
    print(f'  {name}: smallest printed text {sm:.1f} pt ({who!r})'); NUM.setdefault('minfont',{})[name]=round(sm,1)
def done(fig,name,printed_in):
    fontchk(fig,name,printed_in); save(fig,name)
LBC=ListedColormap(['#E3EEFB','#FDEBD2'])
X=np.array([[5,3],[6,2],[3,4],[4,1],[7,5],[8,8],[6,9],[9,8]],float); y=np.array([0,0,0,0,1,1,1,1]); q=np.array([6.,5.])
def fruit(ax,s=46,lab=True):
    ax.scatter(X[y==0,0],X[y==0,1],s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,marker='o',label='apricot')
    ax.scatter(X[y==1,0],X[y==1,1],s=s,color=OR,edgecolor='white',lw=0.8,zorder=5,marker='s',label='plum')
    if lab:
        for i,(a,b) in enumerate(X): ax.text(a+0.22,b-0.52,str(i+1),fontsize=8.5,color=GREY,zorder=6)
def qstar(ax,s=130): ax.scatter([q[0]],[q[1]],s=s,marker='*',color=NAVY,edgecolor='white',lw=0.8,zorder=7)
def fruitaxes(ax,xl='weight (units of 10 g)',yl='colour score'):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_xlim(1.5,10.5); ax.set_ylim(0,10.5)

# ---- 20.0-1 opener: the whole idea
if want('f200.png'):
    fig,ax=plt.subplots(figsize=(3.6,2.75),dpi=240)
    d=np.sqrt(((X-q)**2).sum(1)); o=np.argsort(d)[:3]
    ax.add_patch(Circle(q,d[o[2]],fc='#F1EAF7',ec=PUR,lw=1.3,ls='--',alpha=0.7,zorder=1))
    for i in o: ax.plot([q[0],X[i,0]],[q[1],X[i,1]],color=PUR,lw=1.0,zorder=3)
    fruit(ax,s=55,lab=False); qstar(ax,170)
    ax.text(q[0]-0.15,q[1]+0.45,'query',color=NAVY,fontsize=9,ha='right',weight='bold')
    ax.text(1.3,10.3,'the 3 nearest decide:\n2 apricots, 1 plum',color=PUR,fontsize=9,ha='left',va='top')
    ax.legend(loc='lower right',fontsize=8.5,handletextpad=0.3,borderaxespad=0.2)
    fruitaxes(ax); ax.set_aspect('equal'); ax.set_xlim(1,10.5); ax.set_ylim(0,10.6)
    done(fig,'f200.png',3.65)

# ---- 20.2-1 Voronoi cells and the 3-NN rule
if want('f221.png'):
    fig,axs=plt.subplots(1,2,figsize=(5.0,2.55),dpi=240,sharey=True)
    gx,gy=np.meshgrid(np.linspace(1.5,10.5,400),np.linspace(0,10.5,400))
    for ax,k,t in zip(axs,[1,3],['1 neighbour: Voronoi cells','3 neighbours']):
        m=KNeighborsClassifier(k).fit(X,y); Z=m.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],cmap=LBC,zorder=0); ax.contour(gx,gy,Z,levels=[0.5],colors=GREEN,linewidths=1.6,zorder=2)
        fruit(ax,s=34,lab=False); qstar(ax,95); ax.set_title(t,fontsize=9.5,color=NAVY); fruitaxes(ax)
        ax.set_xlabel('weight (10 g)')
        pr=m.predict([q])[0]; ax.text(10.3,0.4,f'query: {["apricot","plum"][pr]}',fontsize=8.5,ha='right',color=NAVY,bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
    axs[1].set_ylabel(''); fig.tight_layout(w_pad=0.6); done(fig,'f221.png',4.9)

# ---- 20.3-1 unit balls and cosine
if want('f231.png'):
    fig,axs=plt.subplots(1,2,figsize=(4.8,2.25),dpi=240)
    ax=axs[0]; th=np.linspace(0,2*np.pi,721)
    for qq,c,ls,lab in [(1,RED,'-','$q=1$'),(2,GREEN,'-','$q=2$'),(4,PUR,'--','$q=4$')]:
        r=(np.abs(np.cos(th))**qq+np.abs(np.sin(th))**qq)**(-1/qq); ax.plot(r*np.cos(th),r*np.sin(th),color=c,lw=1.6,ls=ls,label=lab)
    ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],color=OR,lw=1.6,ls=':',label='$q=\\infty$')
    ax.set_aspect('equal'); ax.set_xlim(-1.45,1.45); ax.set_ylim(-1.45,1.45); ax.legend(fontsize=8,loc='center',frameon=True,framealpha=0.9,edgecolor='none',handlelength=1.4,labelspacing=0.25,borderpad=0.4)
    ax.set_title('points at distance 1 from $\\mathbf{0}$',fontsize=9,color=NAVY); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.grid(color=GRID,lw=0.5)
    ax=axs[1]; A=np.array([1.,1.]); Bp=np.array([4.,4.]); Cp=np.array([4.,1.])
    ax.scatter(*A,s=50,color=GREY,zorder=5,marker='*'); ax.scatter(*Bp,s=40,color=BLUE,zorder=5); ax.scatter(*Cp,s=40,color=OR,marker='s',zorder=5)
    for P_,t,dx,dy in [(A,'A',0.2,-0.5),(Bp,'B',0.2,-0.5),(Cp,'C',0.2,0.25)]: ax.text(P_[0]+dx,P_[1]+dy,t,fontsize=9,color=NAVY)
    ax.plot([0,4.6],[0,4.6],color=GRID,lw=1,zorder=1); ax.plot([A[0],Cp[0]],[A[1],Cp[1]],color=RED,lw=1.1,zorder=2); ax.plot([A[0],Bp[0]],[A[1],Bp[1]],color=RED,lw=1.1,ls=':',zorder=2)
    cosd=lambda a,b:1-a@b/np.linalg.norm(a)/np.linalg.norm(b)
    rows=[('','Euclid.','cosine'),('A→B',f'{np.linalg.norm(Bp-A):.2f}',f'{cosd(A,Bp):.2f}'),('A→C',f'{np.linalg.norm(Cp-A):.2f}',f'{cosd(A,Cp):.2f}')]
    for j,rw in enumerate(rows):
        for xx,tt,ha in zip([0.1,2.4,3.6],rw,['left','right','right']): ax.text(xx,4.75-0.5*j,tt,fontsize=8.5,color=NAVY if j else GREY,ha=ha,va='top',weight='bold' if j==0 else 'normal')
    NUM['cos']={'AB_euc':round(float(np.linalg.norm(Bp-A)),3),'AB_cos':round(float(cosd(A,Bp)),4),'AC_euc':round(float(np.linalg.norm(Cp-A)),3),'AC_cos':round(float(cosd(A,Cp)),4)}
    ax.set_xlim(0,5); ax.set_ylim(0,5); ax.set_aspect('equal'); ax.grid(color=GRID,lw=0.5); ax.set_title('distance or direction?',fontsize=9,color=NAVY); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4])
    fig.tight_layout(w_pad=0.8); done(fig,'f231.png',4.7)

# ---- 20.3-2 kNN regression on the delivery data
if want('f232.png'):
    xd=np.array([1.,2,3,4,5]); yd=np.array([3.,4,6,7,10]); xg=np.linspace(0,7,701)
    fig,axs=plt.subplots(1,2,figsize=(4.8,1.95),dpi=240,sharey=True)
    for ax,k in zip(axs,[1,3]):
        m=KNeighborsRegressor(k).fit(xd[:,None],yd); ax.plot(xg,m.predict(xg[:,None]),color=GREEN,lw=1.7,drawstyle='default',label=f'$k={k}$')
        ax.plot(xg,0.9+1.7*xg,color=GREY,lw=1,ls='--'); ax.scatter(xd,yd,s=30,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
        ax.set_title(f'$k = {k}$',fontsize=9.5,color=NAVY); ax.set_xlabel('distance $x$ (km)'); ax.grid(color=GRID,lw=0.5); ax.set_xlim(0,7); ax.set_ylim(0,13)
    axs[0].set_ylabel('time $y$ (min)'); axs[1].text(6.9,1.2,'dashed: least\nsquares',fontsize=8.5,color=GREY,ha='right'); axs[0].text(6.9,5.2,'flat beyond\n$x$ = 5',fontsize=8.5,color=GREEN,ha='right',va='top')
    fig.tight_layout(w_pad=0.6); done(fig,'f232.png',4.7)

# ---- 20.4-1 curse of dimensionality
if want('f241.png'):
    rng=np.random.default_rng(0); ds=[1,2,3,5,10,20,50,100,200,500,1000]; rat=[]
    for d in ds:
        P=rng.uniform(size=(500,d)); Q=rng.uniform(size=(100,d)); D=np.sqrt(((Q[:,None]-P[None])**2).sum(2)); rat.append(float((D.min(1)/D.max(1)).mean()))
    NUM['ratio']={'d':ds,'min_over_max':[round(v,3) for v in rat]}
    NUM['shell']={str(d):round(1-0.9**d,4) for d in [2,3,10,50,100]}
    # Monte Carlo check of the shell formula in d = 10
    Z=rng.normal(size=(200000,10)); Z/=np.linalg.norm(Z,axis=1,keepdims=True); R=rng.uniform(size=200000)**(1/10); NUM['shell_mc10']=round(float((R>0.9).mean()),4)
    # accuracy against irrelevant features
    ms=[0,1,2,5,10,20,50,100]; acc=[]
    Xm,ym=make_moons(300,noise=0.25,random_state=1); Xt,yt=make_moons(1000,noise=0.25,random_state=2)
    for m in ms:
        a=[]
        for s_ in range(8):
            r=np.random.default_rng(100+s_); A=np.c_[Xm,r.normal(size=(300,m))]; B=np.c_[Xt,r.normal(size=(1000,m))]
            a.append(KNeighborsClassifier(5).fit(A,ym).score(B,yt))
        acc.append((float(np.mean(a)),float(np.std(a))))
    NUM['noisefeat']={'m':ms,'acc':[round(a,3) for a,_ in acc],'sd':[round(b_,3) for _,b_ in acc]}
    fig,axs=plt.subplots(1,2,figsize=(4.8,2.15),dpi=240)
    ax=axs[0]; ax.semilogx(ds,rat,'-o',color=PUR,ms=3.5,lw=1.5); ax.set_ylim(0,1.02); ax.set_xlabel('dimension $d$'); ax.set_ylabel('nearest / farthest distance'); ax.grid(color=GRID,lw=0.5)
    ax.set_title('500 random points',fontsize=9,color=NAVY)
    ax=axs[1]; mu=np.array([a for a,_ in acc]); sd=np.array([b_ for _,b_ in acc]); ax.plot(ms,mu,'-o',color=BLUE,ms=3.5,lw=1.5); ax.fill_between(ms,mu-sd,mu+sd,color=BLUE,alpha=0.18,lw=0)
    ax.axhline(0.5,color=GREY,lw=1,ls='--'); ax.text(98,0.52,'chance',fontsize=8.5,color=GREY,ha='right',va='bottom'); ax.set_ylim(0.45,1.0); ax.set_xlabel('irrelevant features added'); ax.set_ylabel('test accuracy of 5-NN'); ax.grid(color=GRID,lw=0.5)
    ax.set_title('two moons plus noise columns',fontsize=9,color=NAVY)
    fig.tight_layout(w_pad=0.8); done(fig,'f241.png',4.7)

# ---- 20.5-1 KD-tree: boxes, pruning, and how much is examined
def kd_query(tree_arrays,k,qpt):
    data,idx,nd,nb=tree_arrays; nn=len(nd); heap=[]; seen=0; nodes=0
    def mind(i): 
        lo,hi=nb[0,i],nb[1,i]; return float(np.sqrt((np.maximum(np.maximum(lo-qpt,qpt-hi),0)**2).sum()))
    def rec(i):
        nonlocal seen,nodes
        nodes+=1
        if len(heap)==k and mind(i)>=-heap[0][0]: return
        if 2*i+1>=nn or nd[i]['is_leaf']:
            pts=data[idx[nd[i]['idx_start']:nd[i]['idx_end']]]; dd=np.sqrt(((pts-qpt)**2).sum(1)); seen+=len(dd)
            for dv,ii in zip(dd,idx[nd[i]['idx_start']:nd[i]['idx_end']]):
                if len(heap)<k: heapq.heappush(heap,(-dv,ii))
                elif dv<-heap[0][0]: heapq.heapreplace(heap,(-dv,ii))
            return
        c=[2*i+1,2*i+2]; c.sort(key=mind)
        for j in c: rec(j)
    rec(0); res=sorted([(-a,b_) for a,b_ in heap]); return np.array([a for a,_ in res]),np.array([b_ for _,b_ in res]),seen,nodes
if want('f251.png'):
    rng=np.random.default_rng(4); P=rng.uniform(size=(48,2)); tr=KDTree(P,leaf_size=2); kdarr=tr.get_arrays(); data,idx,nd,nb=kdarr
    qp=np.array([0.62,0.40]); dist,ii,seen,nodes=kd_query(kdarr,3,qp); d0,i0=tr.query(qp[None],k=3); assert np.allclose(dist,d0[0]) and (ii==i0[0]).all()
    rad=dist[-1]; leaves=[i for i in range(len(nd)) if nd[i]['is_leaf']]
    fig,axs=plt.subplots(1,2,figsize=(5.0,2.4),dpi=240,gridspec_kw={'width_ratios':[1.05,1]})
    ax=axs[0]; nsearch=0
    for i in leaves:
        lo,hi=nb[0,i],nb[1,i]; dm=np.sqrt((np.maximum(np.maximum(lo-qp,qp-hi),0)**2).sum()); hit=dm<rad; nsearch+=hit
        ax.add_patch(Rectangle(lo,*(hi-lo),fc='#F1EAF7' if hit else 'white',ec=PUR if hit else GRID,lw=1.0 if hit else 0.8,zorder=1))
    ax.scatter(P[:,0],P[:,1],s=9,color=BLUE,zorder=4); ax.scatter(*qp,s=90,marker='*',color=NAVY,edgecolor='white',lw=0.6,zorder=7)
    ax.add_patch(Circle(qp,rad,fc='none',ec=GREEN,lw=1.4,zorder=6)); ax.scatter(P[ii,0],P[ii,1],s=22,color=BLUE,edgecolor=GREEN,lw=1.3,zorder=5)
    ax.set_aspect('equal'); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,0.5,1]); ax.set_title(f'{len(leaves)} leaf boxes, {nsearch} searched',fontsize=9,color=NAVY)
    NUM['kd']={'leaves':len(leaves),'searched':int(nsearch),'points_examined':int(seen),'n':48,'radius':round(float(rad),3),'nodes_visited':int(nodes),'n_nodes':len(nd)}
    ax=axs[1]; ds=[1,2,3,5,8,10,15,20,30,50]; fr=[]; r2=np.random.default_rng(5)
    for d in ds:
        Pd=r2.uniform(size=(10000,d)); t=KDTree(Pd,leaf_size=30); a=t.get_arrays(); s_=0
        for qq in r2.uniform(size=(40,d)): s_+=kd_query(a,5,qq)[2]
        fr.append(s_/40/10000)
    NUM['kd_frac']={'d':ds,'frac':[round(v,4) for v in fr]}
    ax.semilogx(ds,fr,'-o',color=PUR,ms=3.5,lw=1.5); ax.set_ylim(0,1.03); ax.set_xlabel('dimension $d$'); ax.set_ylabel('share of points examined'); ax.grid(color=GRID,lw=0.5); ax.set_title('KD-tree, $n = 10\\,000$, $k = 5$',fontsize=9,color=NAVY)
    ax.set_xticks([1,2,5,10,20,50]); ax.set_xticklabels(['1','2','5','10','20','50'])
    fig.tight_layout(w_pad=0.8); done(fig,'f251.png',4.9)

# ---- 20.6-1 NCA against PCA on the scaled wine data
if want('f261.png'):
    Xw,yw=load_wine(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xw,yw,test_size=0.25,random_state=0,stratify=yw)
    sc=StandardScaler().fit(Xtr); A=sc.transform(Xtr); B=sc.transform(Xte)
    pca=PCA(2,random_state=0).fit(A); nca=NeighborhoodComponentsAnalysis(n_components=2,random_state=0).fit(A,ytr)
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.5),dpi=240); NUM['nca']={}
    for ax,(nm,tf) in zip(axs,[('first two principal components',pca),('NCA, two components',nca)]):
        Za,Zb=tf.transform(A),tf.transform(B); acc=KNeighborsClassifier(5).fit(Za,ytr).score(Zb,yte); NUM['nca'][nm]=round(float(acc),4)
        for c_,mk,col in [(0,'o',BLUE),(1,'^',GREEN),(2,'s',OR)]:
            ax.scatter(Za[ytr==c_,0],Za[ytr==c_,1],s=14,marker=mk,color=col,edgecolor='white',lw=0.4,zorder=3)
        ax.set_title(f'{nm}',fontsize=8.5,color=NAVY)
        ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_xlabel(f'component 1\n5-NN test accuracy {acc:.3f}',fontsize=8.5); ax.set_ylabel('component 2')
    fig.tight_layout(w_pad=0.8); done(fig,'f261.png',4.8)

# ---- 20.8-1 flow chart
if want('f281.png'):
    fig,ax=new(4.8,3.9)
    def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=7.9): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
    A_=lambda p,q_: arr(ax,p,q_,c=NAVY,lw=0.9,ms=8)
    cx=1.75
    fb(cx,6.0,3.3,0.5,'fit($\\mathbf{X}$, $y$):  validate, store',fc=LGREY,ec=GREY)
    A_((cx,5.75),(cx,5.5))
    ax.add_patch(Polygon([(cx,5.5),(cx+1.2,5.05),(cx,4.6),(cx-1.2,5.05)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(cx,5.05,'algorithm?',ha='center',va='center',fontsize=7.9,color=NAVY)
    xs=[0.65,1.8,2.95]; labs=['brute:\nkeep $\\mathbf{X}$','kd_tree:\nKD-tree','ball_tree:\nball tree']
    for xx,l in zip(xs,labs): fb(xx,3.85,1.08,0.9,l,fc='#F1EAF7',ec=PUR)
    ax.plot([cx-1.2,xs[0]],[5.05,5.05],color=NAVY,lw=0.9); A_((xs[0],5.05),(xs[0],4.3)); A_((cx,4.6),(xs[1],4.3)); ax.plot([cx+1.2,xs[2]],[5.05,5.05],color=NAVY,lw=0.9); A_((xs[2],5.05),(xs[2],4.3))
    for xx in xs: ax.plot([xx,xx],[3.4,3.0],color=NAVY,lw=0.9)
    ax.plot([xs[0],xs[2]],[3.0,3.0],color=NAVY,lw=0.9); A_((cx,3.0),(cx,2.6))
    fb(cx,2.2,3.3,0.8,'stored data and index\n(used by every query)',fc=LGREY,ec=GREY)
    rx=5.8
    fb(rx,6.0,3.0,0.5,'predict($\\mathbf{x}_0$)',fc=LGREY,ec=GREY)
    fb(rx,5.0,3.0,0.7,'find the $k$ nearest rows\n(or all within radius $r$)',fc='#E3F4EA',ec=GREEN)
    fb(rx,3.95,3.0,0.7,'weights $w_i$:  1  or  $1/d_i$\n(distance 0: only those count)')
    fb(rx,2.9,3.0,0.7,'vote $v_c=\\sum w_i[y_i=c]$\nor mean $\\sum w_i y_i/\\sum w_i$')
    fb(rx,1.85,3.0,0.7,'$P(c)=v_c/\\sum v$,  class = argmax\n(a tie goes to the first class)',fc='#E3F4EA',ec=GREEN)
    A_((rx,5.75),(rx,5.35)); A_((rx,4.65),(rx,4.3)); A_((rx,3.6),(rx,3.25)); A_((rx,2.55),(rx,2.2))
    ax.add_patch(FancyArrowPatch((cx+1.65,2.2),(rx-1.5,4.85),arrowstyle='-|>',mutation_scale=8,color=PUR,lw=0.9,ls=(0,(3,2)),connectionstyle='arc3,rad=-0.25'))
    ax.set_xlim(-0.15,7.4); ax.set_ylim(1.2,6.4); done(fig,'f281.png',4.7)

# ---- moons data for 20.11
def moons():
    Xm,ym=make_moons(300,noise=0.3,random_state=0); return train_test_split(Xm,ym,test_size=0.25,random_state=0,stratify=ym)
if want('f2111.png'):
    Xa,Xb,ya,yb=moons(); gx,gy=np.meshgrid(np.linspace(-1.8,2.8,320),np.linspace(-1.5,2.0,320)); NUM['gallery']={}
    fig,axs=plt.subplots(1,4,figsize=(4.9,1.95),dpi=240,sharex=True,sharey=True)
    for ax,k in zip(axs.ravel(),[1,5,15,50]):
        m=KNeighborsClassifier(k).fit(Xa,ya); Z=m.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
        ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],cmap=LBC,zorder=0); ax.contour(gx,gy,Z,levels=[0.5],colors=GREEN,linewidths=1.3,zorder=2)
        ax.scatter(Xa[ya==0,0],Xa[ya==0,1],s=5,color=BLUE,edgecolor='white',lw=0.25,zorder=4); ax.scatter(Xa[ya==1,0],Xa[ya==1,1],s=5,marker='s',color=OR,edgecolor='white',lw=0.25,zorder=4)
        tr,te=m.score(Xa,ya),m.score(Xb,yb); NUM['gallery'][str(k)]={'train':round(float(tr),4),'test':round(float(te),4)}
        ax.set_title(f'$k = {k}$\ntest {te:.2f}',fontsize=8.6,color=NAVY,linespacing=1.15)
        ax.set_xlim(-1.8,2.8); ax.set_ylim(-1.5,2.0); ax.set_xticks([0,2]); ax.set_yticks([-1,0,1,2]); ax.tick_params(labelsize=8); ax.set_xlabel('$x_1$',fontsize=8.6)
    axs[0].set_ylabel('$x_2$',fontsize=8.6)
    fig.tight_layout(w_pad=0.3); done(fig,'f2111.png',4.9)
if want('f2112.png'):
    Xa,Xb,ya,yb=moons(); ks=np.arange(1,61)
    tr,cv=validation_curve(KNeighborsClassifier(),Xa,ya,param_name='n_neighbors',param_range=ks,cv=5)
    cm=cv.mean(1); cs=cv.std(1); kb=int(ks[np.argmax(cm)]); m=KNeighborsClassifier(kb).fit(Xa,ya)
    NUM['valcurve']={'best_k':kb,'cv_best':round(float(cm.max()),4),'test_best':round(float(m.score(Xb,yb)),4),'cv_k1':round(float(cm[0]),4),'cv_k50':round(float(cm[49]),4),'cv_k60':round(float(cm[59]),4),'train_k1':round(float(tr.mean(1)[0]),4),
       'ks_within_1se':[int(k) for k in ks[cm>=cm.max()-cs[np.argmax(cm)]/np.sqrt(5)]][:1]+[int(k) for k in ks[cm>=cm.max()-cs[np.argmax(cm)]/np.sqrt(5)]][-1:]}
    fig,ax=plt.subplots(figsize=(4.2,2.55),dpi=240)
    ax.plot(ks,tr.mean(1),color=BLUE,lw=1.5,label='training folds'); ax.plot(ks,cm,color=GREY,lw=1.5,ls='--',label='validation folds'); ax.fill_between(ks,cm-cs,cm+cs,color=GREY,alpha=0.18,lw=0)
    ax.axvline(kb,color=PUR,lw=1.1); ax.text(kb+1,0.67,f'best $k$ = {kb}',color=PUR,fontsize=8.5,va='bottom')
    ax.set_xlabel('number of neighbours $k$  (flexible $\\leftarrow$ $\\rightarrow$ smooth)'); ax.set_ylabel('accuracy'); ax.set_ylim(0.65,1.01); ax.grid(color=GRID,lw=0.5); ax.legend(loc='lower right',fontsize=8.3,handlelength=1.6)
    done(fig,'f2112.png',4.2)

# ---- 20.12-1 Cover-Hart
if want('f2121.png'):
    from scipy.stats import norm
    from scipy.integrate import quad
    dl=np.array([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]); Rs=norm.cdf(-dl/2)
    def asym(dv):
        eta=lambda x:1/(1+np.exp(-dv*x)); f=lambda x:0.5*norm.pdf(x,dv/2)+0.5*norm.pdf(x,-dv/2)
        return quad(lambda x:2*eta(x)*(1-eta(x))*f(x),-12,12)[0]
    Ra=np.array([asym(v) for v in dl]); sim=[]; rng=np.random.default_rng(0)
    for dv in dl:
        e=[]
        for _ in range(5):
            n=1000; yy=rng.integers(0,2,n); XX=rng.normal(size=(n,2)); XX[:,0]+=np.where(yy==1,dv/2,-dv/2)
            nt=10000; yt=rng.integers(0,2,nt); XT=rng.normal(size=(nt,2)); XT[:,0]+=np.where(yt==1,dv/2,-dv/2)
            e.append(1-KNeighborsClassifier(1).fit(XX,yy).score(XT,yt))
        sim.append(float(np.mean(e)))
    NUM['coverhart']={'delta':dl.tolist(),'bayes':[round(float(v),4) for v in Rs],'asym_1nn':[round(float(v),4) for v in Ra],'bound':[round(float(2*v*(1-v)),4) for v in Rs],'sim_n1000':[round(v,4) for v in sim]}
    fig,ax=plt.subplots(figsize=(3.9,2.9),dpi=240); xx=np.linspace(0,0.5,200)
    ax.plot(xx,xx,color=GREEN,lw=1.4,label='Bayes error $R^*$'); ax.plot(xx,2*xx*(1-xx),color=RED,lw=1.4,ls='--',label='bound $2R^*(1-R^*)$')
    ax.plot(Rs,Ra,'-',color=PUR,lw=1.6,label='1-NN, $n\\to\\infty$'); ax.scatter(Rs,sim,s=20,color=BLUE,zorder=5,label='1-NN, $n$ = 1000')
    ax.set_xlabel('Bayes error $R^*$'); ax.set_ylabel('error rate'); ax.set_xlim(0,0.42); ax.set_ylim(0,0.5); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8.3,loc='upper left',handlelength=1.8)
    done(fig,'f2121.png',3.9)

# ---- 20.13-1 case study: k and scaling on wine
if want('f2131.png'):
    Xw,yw=load_wine(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xw,yw,test_size=0.25,random_state=0,stratify=yw); ks=np.arange(1,26)
    fig,ax=plt.subplots(figsize=(4.2,2.6),dpi=240); NUM['wine']={}
    for nm,mk,col,ls in [('raw features',lambda k:KNeighborsClassifier(k),RED,'-'),('standardised',lambda k:make_pipeline(StandardScaler(),KNeighborsClassifier(k)),GREEN,'-')]:
        cvk=[cross_val_score(mk(k),Xtr,ytr,cv=5).mean() for k in ks]; NUM['wine'][nm]=[round(float(v),4) for v in cvk]
        ax.plot(ks,cvk,color=col,lw=1.6,ls=ls,marker='o',ms=2.8,label=nm)
    ax.set_xlabel('number of neighbours $k$'); ax.set_ylabel('5-fold accuracy'); ax.set_ylim(0.6,1.0); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8.5,loc='center right',handlelength=1.8)
    done(fig,'f2131.png',4.2)

json.dump(NUM,open(NF,'w'),indent=1); print('ok')

# All figures of Chapter 2. Run: python3 fig/figs.py   (PNGs into ch02/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Polygon, Arc, Wedge
NUM={}
only=set(sys.argv[1:])
def want(k): return (not only) or (k in only)
FS=8.2
def A2(ax,p,q,c=BLUE,lw=1.5,ms=8,z=4,ls='-'): 
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=ms,color=c,lw=lw,zorder=z,shrinkA=0,shrinkB=0,ls=ls))
def plane(ax,xl=(-1,5),yl=(-1,5),grid=True,ticks=True):
    ax.set_aspect('equal'); ax.set_xlim(*xl); ax.set_ylim(*yl)
    for s in ('left','bottom'): ax.spines[s].set_visible(True)
    if grid: ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    ax.axhline(0,color=GREY,lw=0.6,zorder=1); ax.axvline(0,color=GREY,lw=0.6,zorder=1)
    ax.tick_params(labelsize=7.5,length=2,pad=1.5)
    if ticks: ax.set_xticks(range(int(np.ceil(xl[0])),int(xl[1])+1)); ax.set_yticks(range(int(np.ceil(yl[0])),int(yl[1])+1))
def T(ax,x,y,s,c=NAVY,fs=FS,**k): ax.text(x,y,s,color=c,fontsize=fs,**k)
def rtangle(ax,o,d1,d2,s=0.18,c=RED):
    o=np.array(o,float); d1=np.array(d1,float)/np.linalg.norm(d1); d2=np.array(d2,float)/np.linalg.norm(d2)
    ax.plot(*zip(o+s*d1,o+s*d1+s*d2,o+s*d2),color=c,lw=0.9,zorder=6)

u=np.array([3.,1.]); v=np.array([1.,2.])

# ---------------------------------------------------------------- f200 opener
if want('f200'):
    fig,axs=plt.subplots(1,3,figsize=(5.1,1.85),dpi=240)
    ax=axs[0]; ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.4,4.8); ax.set_ylim(-0.4,3.6)
    A2(ax,(0,0),u,BLUE); A2(ax,(0,0),v,OR); A2(ax,(0,0),u+v,GREEN,lw=1.8)
    ax.plot([u[0],u[0]+v[0]],[u[1],u[1]+v[1]],color=OR,lw=0.8,ls='--'); ax.plot([v[0],v[0]+u[0]],[v[1],v[1]+u[1]],color=BLUE,lw=0.8,ls='--')
    T(ax,1.6,-0.1,'$\\mathbf{u}$',BLUE,ha='center',va='top'); T(ax,0.35,1.4,'$\\mathbf{v}$',OR,ha='right'); T(ax,2.4,2.0,'$\\mathbf{u}+\\mathbf{v}$',GREEN,ha='left')
    ax.set_title('arrows and sums',fontsize=9,color=NAVY)
    ax=axs[1]; ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-0.4,4.8); ax.set_ylim(-0.4,3.6)
    bb=v.copy(); p=(bb@u)/(u@u)*u
    xx=np.array([-0.3,4.5]); ax.plot(xx,xx*u[1]/u[0],color=GREEN,lw=0.8,ls=':'); A2(ax,(0,0),bb,OR); A2(ax,(0,0),p,GREEN,lw=2.4)
    ax.plot([p[0],bb[0]],[p[1],bb[1]],color=RED,lw=1.3); rtangle(ax,p,bb-p,-p,0.2)
    T(ax,3.9,1.45,'$\\mathbf{u}$',BLUE,ha='left'); T(ax,0.35,1.2,'$\\mathbf{b}$',OR,ha='right'); T(ax,0.8,0.1,'$\\mathbf{p}$',GREEN,ha='center',va='top'); T(ax,1.45,1.3,'$\\mathbf{r}$',RED,ha='left')
    ax.set_title('projection',fontsize=9,color=NAVY)
    ax=axs[2]; ax.axis('off'); ax.set_aspect('equal'); M=np.array([[2.,1.],[-1.,1.]]); Uu,s_,Vt=np.linalg.svd(M); th=np.linspace(0,2*np.pi,200); C=np.c_[np.cos(th),np.sin(th)]
    ax.plot(C[:,0]*0.9,C[:,1]*0.9,color=GREY,lw=0.8,ls='--'); E=C@M.T; ax.plot(E[:,0],E[:,1],color=NAVY,lw=1.3)
    for k,c in enumerate([GREEN,PUR]): A2(ax,(0,0),s_[k]*Uu[:,k],c,lw=1.8)
    ax.set_xlim(-2.9,2.9); ax.set_ylim(-2.2,2.4)
    d1=s_[0]*Uu[:,0]; d2=s_[1]*Uu[:,1]
    T(ax,d1[0]-0.05,d1[1]-0.3,'$\\sigma_1\\mathbf{u}_1$',GREEN,ha='right',va='top'); T(ax,d2[0]+0.1,d2[1]+0.1,'$\\sigma_2\\mathbf{u}_2$',PUR,ha='left',va='bottom')
    ax.set_title('circle $\\to$ ellipse',fontsize=9,color=NAVY)
    fig.tight_layout(w_pad=0.4); save(fig,'f200.png')

# ---------------------------------------------------------------- f211 vectors
if want('f211'):
    fig,axs=plt.subplots(1,3,figsize=(5.1,1.9),dpi=240)
    ax=axs[0]; plane(ax,(-0.5,5.2),(-0.5,3.5)); A2(ax,(0,0),(3,2),BLUE,lw=1.8)
    ax.plot([3,3],[0,2],color=GREY,lw=0.8,ls='--'); ax.plot([0,3],[2,2],color=GREY,lw=0.8,ls='--'); ax.plot(3,2,'o',color=BLUE,ms=3.5)
    T(ax,3.1,0.15,'3',GREY); T(ax,0.1,2.12,'2',GREY); T(ax,3.15,2.55,'$\\mathbf{x}=(3,\\,2)$',BLUE,ha='center'); ax.set_title('arrow and list',fontsize=9,color=NAVY)
    ax=axs[1]; plane(ax,(-0.5,5.2),(-0.5,3.9)); A2(ax,(0,0),u,BLUE); A2(ax,(0,0),v,OR); A2(ax,(0,0),u+v,GREEN,lw=1.8)
    A2(ax,u,u+v,OR,lw=0.9,ls='--',ms=6); A2(ax,v,u+v,BLUE,lw=0.9,ls='--',ms=6)
    T(ax,1.7,0.05,'$\\mathbf{u}$',BLUE,va='top'); T(ax,0.4,1.4,'$\\mathbf{v}$',OR,ha='right'); T(ax,2.6,1.7,'$\\mathbf{u}+\\mathbf{v}$',GREEN,ha='left'); ax.set_title('addition',fontsize=9,color=NAVY)
    ax=axs[2]; plane(ax,(-4.2,7.2),(-2.3,3.2),ticks=False); ax.set_xticks([-3,0,3,6]); ax.set_yticks([-2,0,2])
    A2(ax,(0,0),2*u,GREEN,lw=1.4,ms=7); A2(ax,(0,0),u,BLUE,lw=1.8); A2(ax,(0,0),-u,OR,lw=1.4,ms=7); A2(ax,(0,0),0.5*u,PUR,lw=1.2,ms=6)
    T(ax,6.1,2.1,'$2\\mathbf{u}$',GREEN,ha='left'); T(ax,3.1,0.35,'$\\mathbf{u}$',BLUE,ha='left',va='bottom'); T(ax,-3.1,-1.3,'$-\\mathbf{u}$',OR,ha='right'); T(ax,0.8,-0.35,'$0.5\\mathbf{u}$',PUR,ha='center',va='top',fs=7.5)
    ax.set_title('scaling',fontsize=9,color=NAVY)
    fig.tight_layout(w_pad=0.5); save(fig,'f211.png')

# ---------------------------------------------------------------- f221 norm balls
if want('f221'):
    fig,axs=plt.subplots(1,4,figsize=(5.1,1.55),dpi=240)
    th=np.linspace(0,2*np.pi,400); circ=np.c_[np.cos(th),np.sin(th)]
    dia=np.array([[1,0],[0,1],[-1,0],[0,-1],[1,0]]); sq=np.array([[1,1],[-1,1],[-1,-1],[1,-1],[1,1]])
    pt=np.array([0.6,0.8]); NUM['norms_pt']=[float(np.linalg.norm(pt,p)) for p in (1,2,np.inf)]
    for ax,(t,sh,c,hat) in zip(axs[:3],[('$\\|x\\|_1\\leq 1$',dia,OR,'//'),('$\\|x\\|_2\\leq 1$',circ,BLUE,''),('$\\|x\\|_\\infty\\leq 1$',sq,GREEN,'\\\\')]):
        plane(ax,(-1.5,1.5),(-1.5,1.5),ticks=False); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
        ax.add_patch(Polygon(sh,fc=c,alpha=0.22,ec=c,lw=1.6,hatch=hat,zorder=2)); ax.set_title(t,fontsize=8.8,color=NAVY)
    ax=axs[3]; plane(ax,(-1.5,1.5),(-1.6,1.5),ticks=False); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    ax.plot(dia[:,0],dia[:,1],color=OR,lw=1.5,ls='--'); ax.plot(circ[:,0],circ[:,1],color=BLUE,lw=1.5); ax.plot(sq[:,0],sq[:,1],color=GREEN,lw=1.5,ls=':')
    ax.plot(*pt,'o',color=RED,ms=4,zorder=6); T(ax,0,-1.4,'red point (0.6, 0.8)',RED,fs=7.4,ha='center'); ax.set_title('nested',fontsize=8.8,color=NAVY)
    fig.tight_layout(w_pad=0.3); save(fig,'f221.png')

# ---------------------------------------------------------------- f231 dot product and angle
if want('f231'):
    fig,axs=plt.subplots(1,2,figsize=(5.1,2.25),dpi=240,gridspec_kw={'width_ratios':[1,1]})
    ax=axs[0]; plane(ax,(-0.4,4.0),(-0.4,3.0)); A2(ax,(0,0),u,BLUE,lw=1.8); A2(ax,(0,0),v,OR,lw=1.8)
    p=(v@u)/(u@u)*u; ax.plot([v[0],p[0]],[v[1],p[1]],color=GREY,lw=0.8,ls='--'); ax.plot([0,p[0]],[-0.12,p[1]-0.12],color=GREEN,lw=3,solid_capstyle='butt',zorder=3)
    ax.add_patch(Arc((0,0),1.5,1.5,theta1=np.degrees(np.arctan2(1,3)),theta2=np.degrees(np.arctan2(2,1)),color=PUR,lw=1.2))
    T(ax,0.82,0.6,'$\\theta$',PUR,ha='left',fs=8.4); T(ax,3.1,0.55,'$\\mathbf{u}$',BLUE,ha='left'); T(ax,0.4,1.75,'$\\mathbf{v}$',OR,ha='right'); T(ax,0.9,-0.3,'shadow of $\\mathbf{v}$ on $\\mathbf{u}$',GREEN,fs=7.6,ha='left',va='top')
    ax.set_ylim(-0.75,3.0)
    ax.set_title('$\\mathbf{u}\\cdot\\mathbf{v}=\\|\\mathbf{u}\\|\\,\\|\\mathbf{v}\\|\\cos\\theta$',fontsize=8.8,color=NAVY)
    ax=axs[1]; plane(ax,(-1.95,1.95),(-0.25,1.7),ticks=False); ax.set_xticks([-1,0,1]); ax.set_yticks([0,1])
    ax.plot(np.cos(np.linspace(0,np.pi,100)),np.sin(np.linspace(0,np.pi,100)),color=GRID,lw=1)
    A2(ax,(0,0),(1,0),BLUE,lw=1.8)
    cols={0:BLUE,45:GREEN,90:GREY,135:OR,180:RED}
    for a in (0,45,90,135,180):
        t=np.radians(a); x,y=np.cos(t),np.sin(t); ax.plot([x,x],[0,y],color=cols[a],lw=0.8,ls=':'); ax.plot(x,y,'o',color=cols[a],ms=4,zorder=5)
        ha='left' if x>0.3 else ('right' if x<-0.3 else 'center'); va='bottom' if a in (0,90,180) else 'center'
        T(ax,x*1.14+(0.04 if x>0.3 else (-0.04 if x<-0.3 else 0)),y*1.14+(0.04 if a in (0,180) else 0),f'{a}$^\\circ$: {np.cos(t):.2f}'.replace('-','$-$'),cols[a],fs=7.5,ha=ha,va=va)
    ax.set_title('cosine = shadow of a unit vector',fontsize=8.8,color=NAVY)
    fig.tight_layout(w_pad=0.5); save(fig,'f231.png')

# ---------------------------------------------------------------- f241 projection onto a line and onto a plane
if want('f241'):
    fig=plt.figure(figsize=(5.1,2.4),dpi=240)
    ax=fig.add_axes([0.05,0.14,0.40,0.74]); plane(ax,(-0.5,4.2),(-0.5,2.8))
    xx=np.array([-0.4,4.0]); ax.plot(xx,xx*u[1]/u[0],color=GREEN,lw=0.8,ls=':')
    bb=v.copy(); p=(bb@u)/(u@u)*u
    A2(ax,(0,0),u,BLUE,lw=1.4,ms=7); A2(ax,(0,0),bb,OR,lw=1.8); A2(ax,(0,0),p,GREEN,lw=2.6)
    A2(ax,p,bb,RED,lw=1.5); rtangle(ax,p,bb-p,-p,0.22)
    T(ax,3.1,1.25,'$\\mathbf{u}$',BLUE,ha='left'); T(ax,0.35,1.3,'$\\mathbf{b}$',OR,ha='right'); T(ax,1.2,0.05,'$\\mathbf{p}=c\\mathbf{u}$',GREEN,ha='center',va='top'); T(ax,1.4,1.4,'$\\mathbf{r}$',RED,ha='left')
    ax.set_title('onto a line',fontsize=9,color=NAVY)
    # oblique drawing of a plane in 3-space; screen = (x . e1, x . e2)
    a1=np.array([2.,1,0]); a2=np.array([1.,0,1]); b3=np.array([3.,3,3]); p3=np.array([4.,1,2]); nn=np.cross(a1,a2); nn/=np.linalg.norm(nn)
    e1=a1/np.linalg.norm(a1); ein=np.cross(nn,e1); phi=np.radians(52); e2=np.cos(phi)*nn*np.sign(nn@(b3-p3))+np.sin(phi)*ein
    S3=lambda q: np.array([q@e1,q@e2])
    ax3=fig.add_axes([0.46,0.02,0.54,0.88]); ax3.axis('off'); ax3.set_aspect('equal')
    corners=[a1*s_+a2*t_ for s_,t_ in [(-0.4,-0.4),(1.7,-0.4),(1.7,2.4),(-0.4,2.4)]]
    ax3.add_patch(Polygon([S3(q) for q in corners],fc=LB,ec=BLUE,lw=0.9,alpha=0.9,zorder=1))
    def arr3(q,c,lw,ms=7): A2(ax3,tuple(S3(np.zeros(3))),tuple(S3(q)),c,lw=lw,ms=ms,z=4)
    arr3(a1,BLUE,1.5); arr3(a2,OR,1.5); arr3(p3,GREEN,2.4); arr3(b3,NAVY,1.6)
    A2(ax3,tuple(S3(p3)),tuple(S3(b3)),RED,lw=1.6,ms=7,z=5)
    # right-angle mark at p between r and the in-plane direction towards the origin
    rd=(b3-p3)/np.linalg.norm(b3-p3); pd=-p3/np.linalg.norm(p3); sq=0.35
    ax3.plot(*zip(S3(p3+sq*pd),S3(p3+sq*pd+sq*rd),S3(p3+sq*rd)),color=RED,lw=0.9,zorder=6)
    off=lambda q,d: S3(q)+np.array(d)
    T(ax3,*off(a1,(0.12,-0.05)),'$\\mathbf{a}_1$',BLUE,ha='left',va='top',fs=8.6); T(ax3,*off(a2,(-0.1,0.1)),'$\\mathbf{a}_2$',OR,ha='right',va='bottom',fs=8.6)
    T(ax3,*off(b3,(-0.1,0.08)),'$\\mathbf{b}$',NAVY,ha='right',va='bottom',fs=8.8); T(ax3,*off(p3,(0.12,-0.1)),'$\\mathbf{p}$',GREEN,ha='left',va='top',fs=8.8)
    T(ax3,*off((p3+b3)/2,(0.12,0.0)),'$\\mathbf{r}$',RED,ha='left',va='center',fs=8.8)
    T(ax3,*(S3(corners[3])+np.array([0.3,-0.12])),'plane of all $\\mathbf{A}\\mathbf{c}$',BLUE,fs=7.8,ha='left',va='top')
    allp=np.array([S3(q) for q in corners+[b3,p3]]); ax3.set_xlim(allp[:,0].min()-0.5,allp[:,0].max()+0.6); ax3.set_ylim(allp[:,1].min()-0.5,allp[:,1].max()+0.4)
    fig.text(0.74,0.93,'onto a plane',ha='center',fontsize=9,color=NAVY)
    save(fig,'f241.png')

# ---------------------------------------------------------------- f251 grid deformation
if want('f251'):
    mats=[('scale',np.array([[2,0],[0,0.75]])),('rotate $30^\\circ$',np.array([[np.cos(np.pi/6),-np.sin(np.pi/6)],[np.sin(np.pi/6),np.cos(np.pi/6)]])),
          ('shear',np.array([[1,1],[0,1]])),('project',np.array([[0.5,0.5],[0.5,0.5]]))]
    NUM['f251_det']=[round(float(np.linalg.det(m)),4) for _,m in mats]
    fig,axs=plt.subplots(1,4,figsize=(5.1,1.85),dpi=240)
    for ax,(t,M) in zip(axs,mats):
        plane(ax,(-1.6,3.6),(-1.4,2.6),grid=False,ticks=False); ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-1.6,3.6); ax.set_ylim(-1.6,2.4)
        for k in np.arange(-3,4,1.0):
            for (a,b_) in [((k,-3),(k,3)),((-3,k),(3,k))]:
                P=np.array([a,b_]).T; Q=M@P; ax.plot(Q[0],Q[1],color=GRID if False else '#B9C4D6',lw=0.5,zorder=1)
        sqp=np.array([[0,0],[1,0],[1,1],[0,1]]).T; Q=M@sqp; ax.add_patch(Polygon(Q.T,fc=OR,alpha=0.35,ec=OR,lw=0.8,zorder=2))
        A2(ax,(0,0),tuple(M[:,0]),BLUE,lw=1.8,ms=7); A2(ax,(0,0),tuple(M[:,1]),GREEN,lw=1.8,ms=7)
        ax.set_title(t,fontsize=8.8,color=NAVY); ax.text(0.5,-0.02,f'area $\\times${abs(np.linalg.det(M)):.2f}',transform=ax.transAxes,ha='center',va='top',fontsize=7.8,color=RED)
    fig.tight_layout(w_pad=0.2); save(fig,'f251.png')

# ---------------------------------------------------------------- f252 shapes of a product
if want('f252'):
    fig,ax=plt.subplots(figsize=(5.1,1.9),dpi=240); ax.axis('off'); ax.set_aspect('equal')
    def grid(x0,y0,nr,nc,c=BLUE,fc=LB,hrow=None,hcol=None,cs=0.3):
        for i in range(nr):
            for j in range(nc):
                hl=(hrow==i and hcol is None) or (hcol==j and hrow is None) or (hrow==i and hcol==j)
                ax.add_patch(Rectangle((x0+j*cs,y0-(i+1)*cs),cs,cs,fc=(PINK if hl and c==RED else ('#FDE9C9' if hl and c==OR else ('#CDEBD8' if hl else fc))),ec=c,lw=0.8))
    cs=0.3; y0=2.0
    grid(0.0,y0,4,3,BLUE,LB,hrow=2); grid(2.0,y0,3,2,OR,'#FEF3E0',hcol=1); grid(3.4,y0,4,2,GREEN,'#E3F4EA',hrow=2,hcol=1)
    ax.add_patch(Rectangle((3.4+0.3,y0-3*cs),cs,cs,fc='#7CC49A',ec=GREEN,lw=1.0))
    ax.add_patch(Rectangle((0.0,y0-3*cs),3*cs,cs,fc='none',ec=BLUE,lw=1.6)); ax.add_patch(Rectangle((2.0+cs,y0-3*cs),cs,3*cs,fc='none',ec=OR,lw=1.6))
    T(ax,1.95-0.12,y0-1.5*cs,'$\\times$',NAVY,fs=13,ha='right',va='center'); T(ax,3.35-0.1,y0-1.5*cs,'$=$',NAVY,fs=13,ha='right',va='center')
    T(ax,0.45,y0+0.1,'$\\mathbf{A}$',BLUE,fs=10,ha='center'); T(ax,2.3,y0+0.1,'$\\mathbf{B}$',OR,fs=10,ha='center'); T(ax,3.7,y0+0.1,'$\\mathbf{AB}$',GREEN,fs=10,ha='center')
    T(ax,0.45,y0-4*cs-0.2,'$4\\times 3$',BLUE,ha='center'); T(ax,2.3,y0-4*cs-0.2,'$3\\times 2$',OR,ha='center'); T(ax,3.7,y0-4*cs-0.2,'$4\\times 2$',GREEN,ha='center')
    ax.annotate('',xy=(0.0,y0-4*cs-0.45),xytext=(2.0+2*cs,y0-4*cs-0.45),arrowprops=dict(arrowstyle='<->',color=PUR,lw=0.0)); 
    ax.plot([0.9,0.9],[y0-4*cs-0.38,y0-4*cs-0.34],color=PUR,lw=0)
    T(ax,1.5,y0-4*cs-0.52,'inner sizes agree (3 and 3); outer sizes give the shape',PUR,fs=7.8,ha='center')
    T(ax,4.15,y0-1.0*cs,'row 3 of $\\mathbf{A}$ times\ncolumn 2 of $\\mathbf{B}$\ngives one entry:\n$\\sum_k a_{3k}\\,b_{k2}$',NAVY,fs=7.8,ha='left',va='top')
    ax.set_xlim(-0.1,6.2); ax.set_ylim(y0-4*cs-0.7,y0+0.35); save(fig,'f252.png')

# ---------------------------------------------------------------- f261 independence / span
if want('f261'):
    fig,axs=plt.subplots(1,3,figsize=(5.1,1.95),dpi=240)
    t=np.array([4.,5.])
    cases=[('independent',np.array([2.,1]),np.array([1.,2])),('dependent',np.array([2.,1]),np.array([1.,0.5])),('nearly dependent',np.array([2.,1]),np.array([2.,1.1]))]
    NUM['f261']={}
    for ax,(ttl,a1,a2) in zip(axs,cases):
        plane(ax,(-3.5,7),(-3,7.5),ticks=False); ax.set_xticks([0,4]); ax.set_yticks([0,4])
        cs_=np.arange(-3,4); 
        for c1 in cs_:
            for c2 in cs_:
                q=c1*a1+c2*a2
                if abs(q[0])<7 and abs(q[1])<7.5: ax.plot(*q,'.',color='#9FB4D6',ms=2.2,zorder=2)
        A2(ax,(0,0),a1,BLUE,lw=1.7,ms=7); A2(ax,(0,0),a2,OR,lw=1.7,ms=7)
        T(ax,a1[0]+0.25,a1[1]-1.0,'$\\mathbf{a}_1$',BLUE,ha='left'); T(ax,a2[0]-0.2,a2[1]+0.35,'$\\mathbf{a}_2$',OR,ha='right') if ttl!='dependent' else T(ax,a2[0]-0.1,a2[1]+0.55,'$\\mathbf{a}_2$',OR,ha='right')
        r_=np.linalg.matrix_rank(np.c_[a1,a2]); 
        ax.plot(*t,'X' if ttl=='dependent' else 'o',color=RED,ms=5.5,zorder=6,mec='white',mew=0.5); T(ax,t[0]+0.35,t[1]+0.1,'$\\mathbf{t}$',RED,ha='left')
        if ttl=='independent': c=np.linalg.solve(np.c_[a1,a2],t); T(ax,-3.3,6.4,f'$\\mathbf{{t}}={c[0]:.0f}\\mathbf{{a}}_1+{c[1]:.0f}\\mathbf{{a}}_2$',NAVY,fs=7.6); NUM['f261']['indep_c']=c.tolist()
        elif ttl=='dependent': T(ax,-3.3,6.4,'$\\mathbf{t}$ cannot be reached',NAVY,fs=7.6)
        else: c=np.linalg.solve(np.c_[a1,a2],t); T(ax,-3.3,6.4,f'$\\mathbf{{t}}={c[0]:.0f}\\mathbf{{a}}_1+{c[1]:.0f}\\mathbf{{a}}_2$',NAVY,fs=7.6); NUM['f261']['near_c']=c.tolist()
        ax.set_title(f'{ttl}: rank {r_}',fontsize=8.8,color=NAVY); NUM['f261'][ttl]=int(r_)
    fig.tight_layout(w_pad=0.3); save(fig,'f261.png')

# ---------------------------------------------------------------- f271 row picture of linear systems
if want('f271'):
    fig,axs=plt.subplots(1,3,figsize=(5.1,1.85),dpi=240)
    xs=np.array([-1,7.])
    def line(ax,a,b_,c,col,lab,ls='-',lw=1.6): 
        ax.plot(xs,(c-a*xs)/b_,color=col,lw=lw,ls=ls); 
    ax=axs[0]; plane(ax,(-0.5,5),(-1.5,6),ticks=False); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4])
    line(ax,4,2,10,BLUE,''); line(ax,3,2,8,OR,'',ls='--'); ax.plot(2,1,'o',color=GREEN,ms=6,mec='white',zorder=6); T(ax,2.2,1.25,'(2, 1)',GREEN,ha='left')
    ax.set_title('one solution',fontsize=8.8,color=NAVY); T(ax,2.3,5.8,'$4x+2y=10$\n$3x+2y=8$',NAVY,fs=7.4,va='top')
    ax=axs[1]; plane(ax,(-0.5,5),(-1.5,6),ticks=False); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4])
    line(ax,2,4,10,BLUE,''); line(ax,1,2,8,OR,'',ls='--'); ax.set_title('no solution',fontsize=8.8,color=NAVY); T(ax,0.1,-1.1,'parallel lines',RED,fs=7.6); T(ax,0.1,5.6,'$2x+4y=10$\n$\\;\\;x+2y=8$',NAVY,fs=7.4,va='top')
    ax=axs[2]; plane(ax,(-0.5,5),(-1.5,6),ticks=False); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4])
    line(ax,2,4,10,BLUE,'',lw=3.0); line(ax,1,2,5,OR,'',ls='--'); ax.set_title('many solutions',fontsize=8.8,color=NAVY); T(ax,0.1,-1.1,'the same line',GREEN,fs=7.6); T(ax,0.1,5.6,'$2x+4y=10$\n$\\;\\;x+2y=5$',NAVY,fs=7.4,va='top')
    fig.tight_layout(w_pad=0.3); save(fig,'f271.png')

# ---------------------------------------------------------------- f272 determinant as area
if want('f272'):
    mats=[('$\\det=2$',np.array([[4,2],[3,2.]])),('$\\det=0$',np.array([[2,4],[1,2.]])),('$\\det=-1$',np.array([[0,1],[1,0.]]))]
    fig,axs=plt.subplots(1,3,figsize=(5.1,1.7),dpi=240)
    for ax,(t,M) in zip(axs,mats):
        plane(ax,(-0.8,6.8),(-0.8,5.6),ticks=False); ax.set_xticks([0,2,4,6]); ax.set_yticks([0,2,4])
        ax.add_patch(Polygon([[0,0],[1,0],[1,1],[0,1]],fc=BLUE,alpha=0.3,ec=BLUE,lw=0.8))
        Q=M@np.array([[0,0],[1,0],[1,1],[0,1]]).T; ax.add_patch(Polygon(Q.T,fc=OR,alpha=0.35,ec=OR,lw=1.0,zorder=2))
        A2(ax,(0,0),tuple(M[:,0]),BLUE,lw=1.5,ms=7); A2(ax,(0,0),tuple(M[:,1]),GREEN,lw=1.5,ms=7); ax.set_title(t,fontsize=8.8,color=NAVY)
    T(axs[0],3.2,1.0,'area 2',RED,fs=7.8,ha='left'); T(axs[1],0.3,3.0,'area 0\n(flat)',RED,fs=7.8,ha='left'); T(axs[2],1.4,0.4,'area 1,\norientation\nflipped',RED,fs=7.6,ha='left',va='bottom')
    fig.tight_layout(w_pad=0.3); save(fig,'f272.png')

# ---------------------------------------------------------------- f281 Gram-Schmidt and the shape of QR
if want('f281'):
    fig=plt.figure(figsize=(5.1,2.2),dpi=240)
    ax=fig.add_axes([0.04,0.13,0.50,0.76]); plane(ax,(-0.6,4.0),(-0.7,3.3))
    a1=np.array([3.,1]); a2=np.array([2.,2.5]); q1=a1/np.linalg.norm(a1); pr=(q1@a2)*q1; rr=a2-pr; q2=rr/np.linalg.norm(rr)
    A2(ax,(0,0),a1,BLUE,lw=1.2,ms=7); A2(ax,(0,0),a2,OR,lw=1.2,ms=7); A2(ax,(0,0),q1*1.0,BLUE,lw=2.6); ax.plot([pr[0],a2[0]],[pr[1],a2[1]],color=RED,lw=1.2)
    A2(ax,(0,0),q2*1.0,OR,lw=2.6); A2(ax,(0,0),pr,GREEN,lw=1.6,ms=6,ls='--'); rtangle(ax,pr,rr,-q1,0.2)
    T(ax,3.1,0.75,'$\\mathbf{a}_1$',BLUE,ha='left'); T(ax,2.1,2.6,'$\\mathbf{a}_2$',OR,ha='left'); T(ax,0.55,0.7,'$\\mathbf{q}_1$',BLUE,ha='center'); T(ax,-0.1,1.15,'$\\mathbf{q}_2$',OR,ha='right'); T(ax,2.4,1.8,'$\\mathbf{r}$',RED,ha='left'); T(ax,1.3,-0.1,'part of $\\mathbf{a}_2$ along $\\mathbf{q}_1$',GREEN,fs=7.6,ha='left',va='top')
    ax.set_title('Gram–Schmidt in the plane',fontsize=9,color=NAVY)
    NUM['gs']={'q1':q1.tolist(),'q2':q2.tolist(),'R':[[float(np.linalg.norm(a1)),float(q1@a2)],[0.0,float(np.linalg.norm(rr))]]}
    ax2=fig.add_axes([0.56,0.08,0.43,0.78]); ax2.axis('off'); ax2.set_aspect('equal'); cs=0.26
    def g(x0,y0,nr,nc,c,fc,tri=False):
        for i in range(nr):
            for j in range(nc):
                low=tri and j<i
                ax2.add_patch(Rectangle((x0+j*cs,y0-(i+1)*cs),cs,cs,fc=('white' if low else fc),ec=c,lw=0.7,ls=':' if low else '-'))
                if low: ax2.text(x0+j*cs+cs/2,y0-(i+.5)*cs,'0',ha='center',va='center',fontsize=7,color=GREY)
    y0=2.05; g(0,y0,5,2,BLUE,LB); g(1.15,y0,5,2,GREEN,'#E3F4EA'); g(1.95,y0-0.0,2,2,OR,'#FEF3E0',tri=True)
    ax2.text(0.26,y0+0.1,'$\\mathbf{A}$',color=BLUE,fontsize=10,ha='center'); ax2.text(1.41,y0+0.1,'$\\mathbf{Q}$',color=GREEN,fontsize=10,ha='center'); ax2.text(2.21,y0+0.1,'$\\mathbf{R}$',color=OR,fontsize=10,ha='center')
    ax2.text(0.8,y0-0.65,'$=$',fontsize=12,ha='center',va='center'); 
    T(ax2,0.26,y0-5*cs-0.22,'$n\\times p$',BLUE,ha='center'); T(ax2,1.41,y0-5*cs-0.22,'$n\\times p$',GREEN,ha='center'); T(ax2,2.21,y0-2*cs-0.22,'$p\\times p$',OR,ha='center')
    T(ax2,1.41,y0-5*cs-0.5,'perpendicular\nunit columns',GREEN,fs=7.6,ha='center',va='top'); T(ax2,2.21,y0-2*cs-0.5,'upper\ntriangular',OR,fs=7.6,ha='center',va='top')
    ax2.set_xlim(-0.1,2.8); ax2.set_ylim(0.0,2.5); ax2.set_title('the shape of A = QR',fontsize=9,color=NAVY,y=0.96)
    save(fig,'f281.png')

# ---------------------------------------------------------------- f291 eigenvectors
if want('f291'):
    S=np.array([[5.,2],[2,2]]); lam,V=np.linalg.eigh(S); lam=lam[::-1]; V=V[:,::-1]
    if V[0,0]<0: V[:,0]*=-1
    if V[1,1]>0: V[:,1]*=-1
    fig,axs=plt.subplots(1,2,figsize=(5.1,2.3),dpi=240)
    ax=axs[0]; plane(ax,(-7.2,7.2),(-4.3,4.3),ticks=False); ax.set_xticks([-6,-3,0,3,6]); ax.set_yticks([-3,0,3])
    th=np.linspace(0,2*np.pi,300); C=np.c_[np.cos(th),np.sin(th)]; E=C@S.T
    ax.plot(C[:,0],C[:,1],color=BLUE,lw=1.3); ax.plot(E[:,0],E[:,1],color=NAVY,lw=1.3)
    for t_ in np.radians([100,135,160,200,230,300,330]):
        x=np.array([np.cos(t_),np.sin(t_)]); y=S@x; A2(ax,tuple(x),tuple(y),'#9AA5B8',lw=0.7,ms=5,z=2)
    for k,c in enumerate([GREEN,PUR]):
        ax.plot([-lam[k]*V[0,k]*1.1,lam[k]*V[0,k]*1.1],[-lam[k]*V[1,k]*1.1,lam[k]*V[1,k]*1.1],color=c,lw=0.7,ls=':')
    A2(ax,(0,0),tuple(V[:,0]),GREEN,lw=2.2,ms=7,z=6); A2(ax,tuple(V[:,0]),tuple(6*V[:,0]),GREEN,lw=2.2,ms=8,z=5,ls='--'); A2(ax,(0,0),tuple(V[:,1]),PUR,lw=2.2,ms=7,z=6)
    T(ax,6*V[0,0]+0.1,6*V[1,0]+0.7,'$\\mathbf{S}\\mathbf{v}_1=6\\,\\mathbf{v}_1$',GREEN,ha='right',fs=7.6); T(ax,V[0,0]-0.05,V[1,0]+0.4,'$\\mathbf{v}_1$',GREEN,ha='right',va='bottom'); T(ax,V[0,1]+0.15,V[1,1]-0.1,'$\\mathbf{v}_2$',PUR,ha='left',va='top')
    T(ax,-6.9,-3.7,'blue circle $\\to$ black ellipse',NAVY,fs=7.5); ax.set_title('directions that keep their line',fontsize=8.8,color=NAVY)
    ax=axs[1]; plane(ax,(-7,7),(-4.5,4.5),ticks=False); ax.set_xticks([-6,-3,0,3,6]); ax.set_yticks([-3,0,3])
    rng=np.random.default_rng(2); Z=rng.multivariate_normal([0,0],S,300); ax.plot(Z[:,0],Z[:,1],'o',color=BLUE,ms=2.2,alpha=0.55,mew=0)
    E2=C@np.diag(np.sqrt(lam))@V.T; ax.plot(E2[:,0],E2[:,1],color=NAVY,lw=1.2,zorder=4)
    for k,c in enumerate([GREEN,PUR]):
        d=np.sqrt(lam[k])*V[:,k]; ax.plot([-d[0],d[0]],[-d[1],d[1]],color=c,lw=2.0,zorder=5)
    ax.annotate('$\\sqrt{\\lambda_1}=2.45$',xy=np.sqrt(lam[0])*V[:,0],xytext=(3.0,-2.6),color=GREEN,fontsize=7.8,bbox=dict(fc='white',ec='none',pad=1.0,alpha=0.85),arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.7),zorder=7)
    ax.annotate('$\\sqrt{\\lambda_2}=1$',xy=np.sqrt(lam[1])*V[:,1],xytext=(-6.2,2.7),color=PUR,fontsize=7.8,bbox=dict(fc='white',ec='none',pad=1.0,alpha=0.85),arrowprops=dict(arrowstyle='-',color=PUR,lw=0.7),zorder=7)
    ax.set_title('cloud with covariance $\\mathbf{S}$',fontsize=8.8,color=NAVY)
    NUM['f291']={'lam':lam.tolist(),'v':V.T.tolist(),'sample_cov':np.cov(Z.T).tolist()}
    fig.tight_layout(w_pad=0.5); save(fig,'f291.png')

# ---------------------------------------------------------------- f2101 SVD circle to ellipse
if want('f2101'):
    M=np.array([[2.,1.],[-1.,1.]]); Uu,s,Vt=np.linalg.svd(M); NUM['f2101']={'sigma':s.tolist(),'det':float(np.linalg.det(M)),'U':Uu.tolist(),'Vt':Vt.tolist()}
    th=np.linspace(0,2*np.pi,300); C=np.c_[np.cos(th),np.sin(th)].T
    steps=[('1. the unit circle,\nwith $\\mathbf{v}_1,\\mathbf{v}_2$',C,np.eye(2),Vt.T),('2. $\\mathbf{V}^{\\mathrm{T}}$ turns\n$\\mathbf{v}_j$ onto the axes',Vt@C,Vt,np.eye(2)),
           ('3. $\\Sigma$ stretches\nby $\\sigma_1,\\sigma_2$',np.diag(s)@Vt@C,np.diag(s)@Vt,np.diag(s)),('4. $\\mathbf{U}$ turns\nto $\\mathbf{u}_1,\\mathbf{u}_2$',M@C,M,Uu@np.diag(s))]
    fig,axs=plt.subplots(1,4,figsize=(5.1,1.75),dpi=240)
    for ax,(t,P,Mx,dirs) in zip(axs,steps):
        plane(ax,(-2.8,2.8),(-2.8,2.8),ticks=False); ax.axis('off'); ax.set_aspect('equal'); ax.set_xlim(-2.8,2.8); ax.set_ylim(-2.8,2.8)
        ax.axhline(0,color=GRID,lw=0.6); ax.axvline(0,color=GRID,lw=0.6)
        ax.plot(P[0],P[1],color=NAVY,lw=1.3)
        if t.startswith('1'):
            for k,c in enumerate([GREEN,PUR]): A2(ax,(0,0),tuple(Vt.T[:,k]),c,lw=2,ms=7)
        elif t.startswith('2'):
            for k,c in enumerate([GREEN,PUR]): A2(ax,(0,0),tuple(np.eye(2)[:,k]),c,lw=2,ms=7)
        elif t.startswith('3'):
            for k,c in enumerate([GREEN,PUR]): A2(ax,(0,0),tuple(s[k]*np.eye(2)[:,k]),c,lw=2,ms=7)
        else:
            for k,c in enumerate([GREEN,PUR]): A2(ax,(0,0),tuple(s[k]*Uu[:,k]),c,lw=2,ms=7)
        ax.set_title(t,fontsize=8.2,color=NAVY)
    T(axs[2],1.1,-0.2,f'$\\sigma_1$={s[0]:.2f}',GREEN,fs=7.6,ha='center',va='top'); T(axs[2],0.12,1.45,f'$\\sigma_2$={s[1]:.2f}',PUR,fs=7.6,ha='left',va='bottom')
    fig.tight_layout(w_pad=0.2); save(fig,'f2101.png')

# ---------------------------------------------------------------- f2102 low-rank approximation of an image
if want('f2102'):
    from sklearn.datasets import load_sample_image
    im=load_sample_image('china.jpg').astype(float).mean(axis=2)[::2,::2]; m,n=im.shape
    Uu,s,Vt=np.linalg.svd(im,full_matrices=False); ks=[2,10,40]; kc=[GREEN,OR,PUR]
    NUM['f2102']={'shape':[m,n],'rank':int(np.linalg.matrix_rank(im)),'ks':ks,'rel_err':[],'store':[],'energy':[],'sigma1':float(s[0]),'sigma_last':float(s[-1])}
    fig=plt.figure(figsize=(5.1,3.3),dpi=240)
    gs=fig.add_gridspec(2,1,height_ratios=[1,1.05],hspace=0.38)
    gt=gs[0].subgridspec(1,4,wspace=0.06); gb=gs[1].subgridspec(1,2,wspace=0.42)
    for j,k in enumerate([None]+ks):
        ax=fig.add_subplot(gt[0,j]); ax.axis('off')
        if k is None: ax.imshow(im,cmap='gray',vmin=0,vmax=255); ax.set_title(f'original, rank {NUM["f2102"]["rank"]}',fontsize=8.4,color=NAVY)
        else:
            Ak=(Uu[:,:k]*s[:k])@Vt[:k]; ax.imshow(np.clip(Ak,0,255),cmap='gray',vmin=0,vmax=255)
            re=float(np.linalg.norm(im-Ak)/np.linalg.norm(im)); st=k*(m+n+1)/(m*n); NUM['f2102']['rel_err'].append(re); NUM['f2102']['store'].append(st); NUM['f2102']['energy'].append(float((s[:k]**2).sum()/(s**2).sum()))
            ax.set_title(f'rank {k}',fontsize=8.4,color=kc[j-1]); ax.text(0.5,-0.04,f'{100*st:.1f}% stored',transform=ax.transAxes,ha='center',va='top',fontsize=7.6,color=GREY)
    ax=fig.add_subplot(gb[0,0]); ax.semilogy(np.arange(1,len(s)+1),s,color=BLUE,lw=1.3); ax.set_xlabel('index $j$',fontsize=8.4); ax.set_ylabel('singular value $\\sigma_j$',fontsize=8.4)
    for k,c in zip(ks,kc): ax.axvline(k,color=c,lw=0.9,ls='--')
    ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=7.5); ax.set_xlim(0,len(s)); ax.set_title('singular values',fontsize=8.8,color=NAVY)
    ax=fig.add_subplot(gb[0,1]); err=np.sqrt(np.maximum(0,1-np.cumsum(s**2)/np.sum(s**2))); ax.semilogy(np.arange(1,len(s)+1),err+1e-12,color=RED,lw=1.3)
    ax.set_xlabel('rank $k$',fontsize=8.4); ax.set_ylabel('relative error',fontsize=8.4); ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=7.5); ax.set_xlim(0,len(s)); ax.set_ylim(1e-3,1)
    for k,c in zip(ks,kc): ax.axvline(k,color=c,lw=0.9,ls='--')
    ax.set_title('error of the best rank-$k$ copy',fontsize=8.8,color=NAVY)
    fig.subplots_adjust(left=0.1,right=0.98,bottom=0.12,top=0.93); save(fig,'f2102.png')

# ---------------------------------------------------------------- f2103 conditioning
if want('f2103'):
    fig,axs=plt.subplots(1,2,figsize=(5.1,2.2),dpi=240)
    ax=axs[0]; plane(ax,(-0.3,8.3),(-0.3,8.6),ticks=False); ax.set_xticks([0,2,4,6,8]); ax.set_yticks([0,2,4,6,8])
    xs=np.array([-1,9.]); a=lambda x:x; b0=lambda x:1.05*x-0.2; b1=lambda x:1.05*x-0.3
    ax.plot(xs,a(xs),color=BLUE,lw=1.5); ax.plot(xs,b0(xs),color=OR,lw=1.5,ls='--'); ax.plot(xs,b1(xs),color=RED,lw=1.3,ls=':')
    x0=0.2/0.05; x1=0.3/0.05; NUM['f2103']={'x0':x0,'x1':x1}
    ax.plot(x0,x0,'o',color=GREEN,ms=4.5,zorder=6); ax.plot(x1,x1,'o',color=RED,ms=4.5,zorder=6)
    T(ax,8.1,0.3,'shift by 0.1:\ncrossing moves\nfrom 4 to 6',NAVY,fs=7.5,ha='right',va='bottom')
    ax.set_title('nearly parallel lines',fontsize=8.8,color=NAVY)
    ax=axs[1]; rho=np.linspace(0,0.9995,400); ax.semilogy(rho,np.sqrt((1+rho)/(1-rho)),color=PUR,lw=1.6)
    for r_,(lab,dx,dy) in zip([0.0,0.9,0.999],[('$\\rho$=0',0.03,1.5),('0.9',-0.02,1.9),('0.999',-0.02,1.9)]):
        cv=np.sqrt((1+r_)/(1-r_)); ax.plot(r_,cv,'o',color=RED,ms=4.5,zorder=5); ax.text(r_+dx,cv*dy,f'{lab}: {cv:.1f}',fontsize=7.6,color=NAVY,ha='left' if r_==0 else 'right')
    ax.set_xlabel('correlation $\\rho$ of two features',fontsize=8.5); ax.set_ylabel('condition number',fontsize=8.5); ax.grid(color=GRID,lw=0.5); ax.tick_params(labelsize=7.5)
    ax.set_title('cond($\\mathbf{X}$) grows as $\\rho\\to 1$',fontsize=8.8,color=NAVY); ax.set_ylim(0.7,200); ax.set_xlim(-0.03,1.02)
    fig.tight_layout(w_pad=0.6); save(fig,'f2103.png')

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)
print('figs ok')

# All main figures of Chapter 17. Run: python3 fig/figs.py   (PNGs into ch17/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import warnings; warnings.filterwarnings('ignore')
from matplotlib.lines import Line2D
NUM={}
SOFTOR='#FDF0E0'
# the four tiles
TX=np.array([[4,3],[1,0],[3,4],[0,1]],float); TY=np.array([1,-1,1,-1])

def perceptron_run(X,y,epochs=50,eta=1.0):
    w=np.zeros(X.shape[1]); b=0.0; hist=[]; 
    for e in range(epochs):
        m=0
        for i in range(len(y)):
            z=y[i]*(X[i]@w+b)
            if z<=0:
                hist.append((e+1,i,w.copy(),b,z)); w=w+eta*y[i]*X[i]; b=b+eta*y[i]; m+=1
        if m==0: break
    return w,b,hist
def boundary(ax,w,b,xlim,ylim,**k):
    # draw w.x+b=0 clipped to the axes
    x0,x1=xlim; y0,y1=ylim; pts_=[]
    if abs(w[1])>1e-12:
        for xx in (x0,x1):
            yy=-(w[0]*xx+b)/w[1]
            if y0-1e-9<=yy<=y1+1e-9: pts_.append((xx,yy))
    if abs(w[0])>1e-12:
        for yy in (y0,y1):
            xx=-(w[1]*yy+b)/w[0]
            if x0-1e-9<=xx<=x1+1e-9: pts_.append((xx,yy))
    if len(pts_)>=2:
        pts_=sorted(set(pts_)); ax.plot([pts_[0][0],pts_[-1][0]],[pts_[0][1],pts_[-1][1]],**k)
def shade(ax,w,b,xl,yl,n=120):
    gx,gy=np.meshgrid(np.linspace(*xl,n),np.linspace(*yl,n)); f=w[0]*gx+w[1]*gy+b
    ax.contourf(gx,gy,f,levels=[-1e9,0,1e9],colors=[LB,SOFTOR],alpha=0.9,zorder=0)
def cls(ax,X,y,s=34,lab=True):
    ax.scatter(X[y<0,0],X[y<0,1],s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,marker='o',label='cracked ($y=-1$)' if lab else None)
    ax.scatter(X[y>0,0],X[y>0,1],s=s,color=OR,edgecolor='white',lw=0.8,zorder=5,marker='s',label='sound ($y=+1$)' if lab else None)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ---- 17.0-1 opener: boundaries of a perceptron run on a cloud
def make_cloud(seed,n1=12,sig=0.65,c=1.4):
    rng=np.random.default_rng(seed)
    X=np.r_[rng.normal([-c,-c*0.55],[sig,sig],(n1,2)),rng.normal([c,c*0.55],[sig,sig],(n1,2))]; y=np.r_[-np.ones(n1),np.ones(n1)]
    p=rng.permutation(2*n1); return X[p],y[p]
seed=21; Xo,yo=make_cloud(seed); wo,bo,ho=perceptron_run(Xo,yo)
print('opener seed',seed,'updates',len(ho))
NUM['opener']={'seed':seed,'n':len(yo),'updates':len(ho)}
fig,ax=plt.subplots(figsize=(5.4,1.5),dpi=240)
xl,yl=(-4,4),(-1.8,2.8)
for k,(e,i,w,b,z) in enumerate(ho):
    if np.linalg.norm(w)<1e-12: continue
    boundary(ax,w,b,xl,yl,color=GREY,lw=0.9,alpha=0.35+0.45*k/len(ho),zorder=2)
boundary(ax,wo,bo,xl,yl,color=GREEN,lw=2.4,zorder=4)
cls(ax,Xo,yo,s=30,lab=False)
ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('feature  $x_1$'); ax.set_ylabel('feature  $x_2$')
ax.set_ylim(-2.0,3.0)
ax.text(-3.9,2.75,'class $-1$',color=BLUE,fontsize=10,va='center',weight='bold'); ax.text(3.9,2.75,'class $+1$',color=OR,fontsize=10,va='center',ha='right',weight='bold')
ax.text(1.35,-1.7,'final boundary',color=GREEN,fontsize=10,va='center',ha='left',weight='bold')
ax.text(-2.4,1.45,'grey: boundaries after\nearlier mistakes',color=GREY,fontsize=9.5,va='center',ha='center',linespacing=1.3)
save(fig,'f1700.png')

# ---- 17.1-1 the four tiles
fig,ax=plt.subplots(figsize=(3.6,2.55),dpi=240); cls(ax,TX,TY,s=52)
for k,(p,yv) in enumerate(zip(TX,TY)):
    dx,dy=(0.14,-0.42) if k in (0,1) else (0.14,0.18)
    if k==2: dx,dy=-0.2,0.2
    ax.text(p[0]+dx,p[1]+dy,f'{k+1}: ({p[0]:.0f}, {p[1]:.0f})',fontsize=9,color=GREY,ha='left' if k!=2 else 'right')
ax.set_xlim(-0.6,5.6); ax.set_ylim(-0.7,5.3); grid(ax); ax.set_xlabel('ring reading $x_1$'); ax.set_ylabel('gloss reading $x_2$')
ax.legend(loc='lower right',fontsize=8.5,handletextpad=0.2,borderaxespad=0.3); save(fig,'f1711.png')

# ---- 17.2-1 one correction: before and after (update 4 of the first pass)
w1,b1=np.array([3.,3.]),0.0; w2,b2=np.array([3.,2.]),-1.0
fig,axs=plt.subplots(1,2,figsize=(5.4,2.55),dpi=240,sharey=True); xl,yl=(-1.0,5.6),(-1.0,5.4)
for ax,(w,b,t) in zip(axs,[(w1,b1,'before: $\\mathbf{w}=(3,3),\\ b=0$'),(w2,b2,'after:  $\\mathbf{w}=(3,2),\\ b=-1$')]):
    shade(ax,w,b,xl,yl); boundary(ax,w,b,xl,yl,color=GREEN,lw=2,zorder=3)
    if ax is axs[1]: boundary(ax,w1,b1,xl,yl,color=GREY,lw=1,ls='--',zorder=2)
    cls(ax,TX,TY,s=40,lab=False)
    ax.add_patch(plt.Circle(TX[3],0.34,fc='none',ec=RED,lw=1.4,zorder=6))
    # normal vector w, drawn from the foot of the perpendicular from the origin (scaled to length 1.4)
    n=w/np.linalg.norm(w); foot=-b*w/(w@w); arr(ax,foot,foot+1.5*n,c=GREEN,lw=1.4,ms=8,zorder=4)
    ax.text(foot[0]+1.5*n[0]+0.12,foot[1]+1.5*n[1]-0.1,'$\\mathbf{w}$',color=GREEN,fontsize=11,va='top')
    ax.set_xlim(*xl); ax.set_ylim(*yl); grid(ax); ax.set_xlabel('$x_1$'); ax.set_title(t,fontsize=9.5,color=NAVY)
    z=TY[3]*(w@TX[3]+b); ax.text(0.15,0.55 if ax is axs[0] else 0.55,'',fontsize=8)
    ax.text(0.45,1.5,f'sample 4: $z={z:+.0f}$',color=RED,fontsize=9,va='bottom')
axs[0].set_ylabel('$x_2$'); fig.tight_layout(w_pad=0.8); save(fig,'f1721.png')

# ---- 17.7-1 the whole perceptron run on the four tiles
w,b,hist=perceptron_run(TX,TY); assert len(hist)==5
fig,axs=plt.subplots(2,3,figsize=(5.4,3.75),dpi=240,sharex=True,sharey=True); xl,yl=(-0.8,5.4),(-0.8,5.4); wt,bt=np.zeros(2),0.0
steps=[]
for (e,i,w0,b0,z0) in hist:
    wt=w0+TY[i]*TX[i]; bt=b0+TY[i]; steps.append((e,i,wt.copy(),bt))
for k,(ax,(e,i,wt,bt)) in enumerate(zip(axs.ravel()[:5],steps)):
    shade(ax,wt,bt,xl,yl); boundary(ax,wt,bt,xl,yl,color=GREEN,lw=1.8,zorder=3); cls(ax,TX,TY,s=26,lab=False)
    ax.add_patch(plt.Circle(TX[i],0.38,fc='none',ec=RED,lw=1.3,zorder=6))
    ax.set_title(f'update {k+1}: $\\mathbf{{w}}=({wt[0]:.0f},\\,{wt[1]:.0f}),\\ b={bt:.0f}$',fontsize=8.8,color=NAVY)
ax=axs.ravel()[5]; shade(ax,w,b,xl,yl); boundary(ax,w,b,xl,yl,color=GREEN,lw=1.8,zorder=3); cls(ax,TX,TY,s=26,lab=False)
for k,p in enumerate(TX):
    zz=TY[k]*(w@p+b); off={0:(0.25,-0.1,'top','left'),1:(-0.15,-0.2,'top','right'),2:(0.25,-0.1,'top','left'),3:(0.25,0.05,'bottom','left')}[k]
    ax.text(p[0]+off[0],p[1]+off[1],f'$z={zz:.0f}$',fontsize=8.8,color=GREY,va=off[2],ha=off[3])
ax.set_title('final: margins $z_i$',fontsize=8.8,color=NAVY)
for ax in axs.ravel(): ax.set_xlim(*xl); ax.set_ylim(*yl); grid(ax); ax.set_xticks([0,2,4]); ax.set_yticks([0,2,4]); ax.tick_params(labelsize=8.5)
for ax in axs[1]: ax.set_xlabel('$x_1$')
for ax in axs[:,0]: ax.set_ylabel('$x_2$')
fig.tight_layout(h_pad=0.6,w_pad=0.4); save(fig,'f1771.png'); NUM['tiles']={'final_w':w.tolist(),'final_b':b,'updates':len(hist),'margins':[float(TY[k]*(w@TX[k]+b)) for k in range(4)]}
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

# ---- 17.4-1 the losses as functions of the margin z, and their slopes
z=np.linspace(-2.5,2.5,1001)
L={'0-1':(z<=0)*1.0,'perceptron':np.maximum(0,-z),'hinge':np.maximum(0,1-z),'squared_hinge':np.maximum(0,1-z)**2,
   'log_loss':np.log1p(np.exp(-z)),'modified_huber':np.where(z>=1,0,np.where(z>=-1,(1-z)**2,-4*z))}
D={'perceptron':-1.0*(z<=0),'hinge':-1.0*(z<=1),'squared_hinge':-2*np.maximum(0,1-z),'log_loss':-1/(1+np.exp(z)),
   'modified_huber':np.where(z>=1,0,np.where(z>=-1,-2*(1-z),-4.0))}
sty={'0-1':(NAVY,'-',1.0),'perceptron':(PUR,':',1.9),'hinge':(BLUE,'-',1.8),'squared_hinge':(OR,'--',1.8),'log_loss':(GREEN,'-',1.8),'modified_huber':(RED,'-.',1.8)}
fig,axs=plt.subplots(1,2,figsize=(5.4,2.65),dpi=240)
for ax,dat,yl_,tt in zip(axs,[L,D],[(-0.1,4.6),(-4.8,0.55)],['loss  $L(z)$','slope  $L\'(z)$']):
    for k,v in dat.items():
        c_,ls_,lw_=sty[k]; ax.plot(z,v,color=c_,ls=ls_,lw=lw_,label=k.replace('_',' ') if k!='0-1' else '0–1',drawstyle='steps-mid' if k=='0-1' else 'default')
    ax.axvline(0,color=GRID,lw=0.8,zorder=0); ax.axhline(0,color=GRID,lw=0.8,zorder=0); ax.set_ylim(*yl_); ax.set_xlim(-2.5,2.5); grid(ax)
    ax.set_xlabel('margin  $z=y\\,f(\\mathbf{x})$'); ax.set_title(tt,fontsize=9.5,color=NAVY)
axs[0].text(-2.4,0.45,'wrong side',fontsize=8.5,color=GREY,ha='left'); axs[0].text(2.4,0.45,'right side',fontsize=8.5,color=GREY,ha='right')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=6,fontsize=8,handlelength=1.8,columnspacing=0.8,handletextpad=0.4,bbox_to_anchor=(0.5,1.01))
fig.tight_layout(rect=(0,0,1,0.92),w_pad=1.0); save(fig,'f1741.png')
NUM['losses']={'z':[-2.0,0.5],'check':'see verify/checks.json'}

# ---- 17.5-1 learning-rate schedules
t=np.logspace(0,5,400); a=1e-4
fig,axs=plt.subplots(1,2,figsize=(5.4,2.65),dpi=240); ax=axs[0]
ax.loglog(t,np.full_like(t,0.01),color=BLUE,lw=1.8,label='constant,  $\\eta_0=0.01$')
ax.loglog(t,0.01/t**0.5,color=OR,lw=1.8,ls='--',label='invscaling,  $p=0.5$')
ax.loglog(t,0.01/t**0.25,color=PUR,lw=1.8,ls=':',label='invscaling,  $p=0.25$')
ax.loglog(t,1/(a**0.25+a*(t-1)),color=GREEN,lw=1.8,ls='-',label='optimal,  $\\alpha=10^{-4}$')
ax.set_xlabel('update number  $t$'); ax.set_ylabel('learning rate  $\\eta_t$'); ax.set_title('the three formulas',fontsize=9.5,color=NAVY); ax.grid(color=GRID,lw=0.5,which='major'); ax.set_ylim(1e-7,60)
ax.legend(fontsize=8,loc='lower left',handlelength=2.2,borderaxespad=0.3,labelspacing=0.3)
ax=axs[1]; ep=np.arange(0,41); eta=np.where(ep<8,0.1,np.where(ep<15,0.02,np.where(ep<22,0.004,0.0008)))
ax.step(ep,eta,where='post',color=RED,lw=1.8); ax.set_yscale('log'); ax.set_xlabel('epoch'); ax.set_ylabel('$\\eta$'); ax.set_title('adaptive (schematic)',fontsize=9.5,color=NAVY); ax.grid(color=GRID,lw=0.5)
ax.annotate('loss stopped improving\nfor 5 epochs: $\\eta\\to\\eta/5$',xy=(8,0.1),xytext=(11,0.09),fontsize=8.3,color=RED,va='center',arrowprops=dict(arrowstyle='-|>',color=RED,lw=0.8))
ax.set_ylim(3e-4,0.3)
fig.tight_layout(w_pad=1.0); save(fig,'f1751.png')

# ---- 17.6-1 ridge classifier = regression on +-1 targets; easy points far away pull the line
from sklearn.linear_model import RidgeClassifier, LogisticRegression
x0=np.array([1,2,3,4,5,6.]); y0=np.array([-1,-1,-1,1,1,1]); far=np.array([12.,14.,16.])
fig,axs=plt.subplots(1,2,figsize=(5.4,2.5),dpi=240,sharey=True); NUM['ridge1d']={}
for ax,(x,y,tt,key) in zip(axs,[(x0,y0,'six readings','six'),(np.r_[x0,far],np.r_[y0,np.ones(3)],'three easy points added','nine')]):
    X1=x[:,None]; r=RidgeClassifier(alpha=1e-6).fit(X1,y); lg=LogisticRegression(C=1e4,max_iter=10000).fit(X1,y)
    bdr=-r.intercept_[0]/r.coef_.ravel()[0]; bdl=-lg.intercept_[0]/lg.coef_.ravel()[0]
    NUM['ridge1d'][key]={'ridge_boundary':round(float(bdr),3),'logistic_boundary':round(float(bdl),3),'ridge_acc':float(r.score(X1,y)),'slope':round(float(r.coef_.ravel()[0]),4)}
    xx=np.array([0,17.5]); ax.plot(xx,r.coef_.ravel()[0]*xx+r.intercept_[0],color=GREEN,lw=1.8,zorder=3)
    ax.axhline(1,color=GRID,lw=0.8,ls=':'); ax.axhline(-1,color=GRID,lw=0.8,ls=':'); ax.axhline(0,color=GREY,lw=0.7)
    ax.axvline(bdr,color=GREEN,lw=1.2,ls='-',zorder=2); ax.axvline(bdl,color=PUR,lw=1.4,ls='--',zorder=2)
    ax.scatter(x[y<0],y[y<0],s=34,color=BLUE,marker='o',edgecolor='white',lw=0.8,zorder=5); ax.scatter(x[y>0],y[y>0],s=34,color=OR,marker='s',edgecolor='white',lw=0.8,zorder=5)
    ax.set_xlim(0,17.5); ax.set_ylim(-2.2,3.2); grid(ax); ax.set_xlabel('reading $x$'); ax.set_title(tt,fontsize=9.5,color=NAVY)
    if key=='six': ax.text(bdr+0.3,-1.75,'both methods: 3.50',color=NAVY,fontsize=8.8,ha='left')
    else: ax.text(bdr+0.3,-1.75,f'ridge: {bdr:.2f}',color=GREEN,fontsize=8.8,ha='left'); ax.text(bdl-0.3,2.6,f'log. regr.: {bdl:.2f}',color=PUR,fontsize=8.8,ha='right')
axs[0].set_ylabel('target $t=\\pm1$ and fitted line'); 
axs[1].annotate('misclassified',xy=(4,1),xytext=(7.3,1.7),fontsize=8.8,color=RED,arrowprops=dict(arrowstyle='-|>',color=RED,lw=0.9),va='center')
fig.tight_layout(w_pad=0.8); save(fig,'f1761.png'); json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

# ---- 17.8-1 flow chart of SGDClassifier.fit
fig,ax=new(4.6,5.9); fs_=8.6
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=fs_): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.05)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
cx=2.3
fb(cx,5.62,4.2,0.42,'Input: $\\mathbf{X}$ ($n\\times p$), $\\mathbf{y}$, and the settings',fc=LGREY,ec=GREY)
fb(cx,4.98,4.2,0.55,'Validate; code the labels as $\\pm1$\n(one binary problem per class if $K>2$)')
fb(cx,4.32,4.2,0.42,'Start:  $\\mathbf{w}=\\mathbf{0}$,  $b=0$,  $t=1$')
ax.add_patch(FancyBboxPatch((0.05,0.72),4.5,3.18,boxstyle='round,pad=0.02,rounding_size=0.06',fc='none',ec=PUR,lw=1.0,ls='--'))
ax.text(0.15,3.78,'repeat once per epoch',fontsize=8.6,color=PUR,va='center',style='italic')
fb(cx,3.4,4.0,0.4,'Shuffle the order of the samples',fc='#F1EAF7',ec=PUR)
fb(cx,2.62,4.0,1.05,'for each sample $i$:   $z=y_i(\\mathbf{w}^{\\mathrm{T}}\\mathbf{x}_i+b)$,   $\\eta_t$ from the schedule\n$\\mathbf{w}\\leftarrow(1-\\eta(1-\\rho)\\alpha)\\mathbf{w}-\\eta L^{\\prime}(z)y_i\\mathbf{x}_i$\n$b\\leftarrow b-\\eta L^{\\prime}(z)y_i$;   $t\\leftarrow t+1$',fc='#E3F4EA',ec=GREEN)
fb(cx,1.78,4.0,0.42,'L1 truncation; running average $\\bar{\\mathbf{w}}$ if requested',fc='#F1EAF7',ec=PUR)
ax.add_patch(Polygon([(cx,1.5),(cx+1.7,1.12),(cx,0.76),(cx-1.7,1.12)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(cx,1.12,'no gain for 5 epochs,\nor max_iter reached?',ha='center',va='center',fontsize=8.4,color=NAVY,linespacing=1.15)
fb(cx,0.3,4.2,0.42,'Store  coef_,  intercept_,  n_iter_,  t_',fc=LGREY,ec=GREY)
A((cx,5.41),(cx,5.26)); A((cx,4.71),(cx,4.53)); A((cx,4.11),(cx,3.9)); A((cx,3.2),(cx,3.15)); A((cx,2.09),(cx,1.99)); A((cx,1.57),(cx,1.51)); A((cx,0.76),(cx,0.52))
ax.text(cx+0.1,0.62,'yes',fontsize=8.6,color=NAVY,va='center')
# 'no' loop back
ax.plot([cx+1.7,4.38,4.38],[1.12,1.12,3.4],color=NAVY,lw=0.9); A((4.38,3.4),(cx+2.0,3.4)); ax.text(cx+1.75,1.2,'no',fontsize=8.6,color=NAVY,va='bottom')
ax.text(cx+1.07,0.62,'',fontsize=8)
ax.text(0.15,0.12,'',fontsize=8)
ax.set_xlim(0,4.6); ax.set_ylim(0.05,5.9); save(fig,'f1781.png')

# ---- 17.11-1 gallery: the strength alpha
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xb,yb=make_blobs(n_samples=70,centers=[(-1.0,-0.7),(1.0,0.7)],cluster_std=1.15,random_state=2); sb=2*yb-1
alphas4=[1e-4,1e-2,1e-1,1.0]; NUM['alpha_gallery']={}
fig,axs=plt.subplots(1,4,figsize=(5.4,1.85),dpi=240,sharex=True,sharey=True); xl,yl=(-3.9,3.9),(-3.4,3.4)
for ax,al in zip(axs,alphas4):
    m=SGDClassifier(loss='hinge',alpha=al,max_iter=2000,tol=None,random_state=0).fit(Xb,yb); w=m.coef_[0]; b=m.intercept_[0]
    shade(ax,w,b,xl,yl); 
    for off,ls_,lw_ in [(1,'--',0.9),(-1,'--',0.9)]: boundary(ax,w,b-off,xl,yl,color=GREEN,lw=lw_,ls=ls_,zorder=3)
    boundary(ax,w,b,xl,yl,color=GREEN,lw=1.8,zorder=4)
    ax.scatter(Xb[yb==0,0],Xb[yb==0,1],s=12,color=BLUE,edgecolor='white',lw=0.5,zorder=5,marker='o'); ax.scatter(Xb[yb==1,0],Xb[yb==1,1],s=12,color=OR,edgecolor='white',lw=0.5,zorder=5,marker='s')
    acc=m.score(Xb,yb); width=2/np.linalg.norm(w); NUM['alpha_gallery'][str(al)]={'norm_w':round(float(np.linalg.norm(w)),3),'margin_width':round(float(width),3),'train_acc':round(float(acc),4)}
    ax.set_title(f'$\\alpha={al:g}$',fontsize=9.5,color=NAVY); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xticks([]); ax.set_yticks([])
    ax.text(-3.75,3.25,f'$\\Vert\\mathbf{{w}}\\Vert={np.linalg.norm(w):.2f}$',fontsize=8.8,color=GREEN,ha='left',va='top',zorder=10,bbox=dict(fc='white',ec='none',alpha=0.85,pad=1.0))
fig.tight_layout(w_pad=0.3); save(fig,'f1811.png')
print(NUM['alpha_gallery'])

# ---- 17.11-2 gallery: the loss
fig,axs=plt.subplots(1,4,figsize=(5.4,1.85),dpi=240,sharex=True,sharey=True); NUM['loss_gallery']={}
for ax,(lo,al) in zip(axs,[('perceptron',1e-4),('hinge',1e-2),('log_loss',1e-2),('modified_huber',1e-2)]):
    kw=dict(penalty=None) if lo=='perceptron' else dict(alpha=al)
    m=SGDClassifier(loss=lo,max_iter=2000,tol=None,random_state=0,**({'learning_rate':'constant','eta0':1.0} if lo=='perceptron' else kw)).fit(Xb,yb); w=m.coef_[0]; b=m.intercept_[0]
    shade(ax,w,b,xl,yl); boundary(ax,w,b,xl,yl,color=GREEN,lw=1.8,zorder=4)
    ax.scatter(Xb[yb==0,0],Xb[yb==0,1],s=12,color=BLUE,edgecolor='white',lw=0.5,zorder=5,marker='o'); ax.scatter(Xb[yb==1,0],Xb[yb==1,1],s=12,color=OR,edgecolor='white',lw=0.5,zorder=5,marker='s')
    ax.set_title(lo.replace('_',' '),fontsize=9.5,color=NAVY); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_xticks([]); ax.set_yticks([])
    acc=m.score(Xb,yb); NUM['loss_gallery'][lo]={'train_acc':round(float(acc),4),'norm_w':round(float(np.linalg.norm(w)),3)}
    ax.text(-3.75,3.25,f'accuracy {acc:.3f}',fontsize=8.8,color=GREY,ha='left',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
fig.tight_layout(w_pad=0.3); plt.close(fig); print(NUM['loss_gallery'])

# ---- 17.11-3 validation curve for alpha, and convergence of the schedules
Xc,yc=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xc,yc,test_size=0.25,stratify=yc,random_state=0)
cv=StratifiedKFold(5,shuffle=True,random_state=0); sc=StandardScaler().fit(Xtr); Xs,Xts=sc.transform(Xtr),sc.transform(Xte)
pipe=make_pipeline(StandardScaler(),SGDClassifier(random_state=0)); pr=np.logspace(-7,1,17)
trs,vas=validation_curve(pipe,Xtr,ytr,param_name='sgdclassifier__alpha',param_range=pr,cv=cv,scoring='accuracy')
fig,axs=plt.subplots(1,2,figsize=(5.4,2.7),dpi=240); ax=axs[0]
ax.semilogx(pr,trs.mean(1),color=BLUE,lw=1.6,marker='o',ms=3.2,label='training'); ax.semilogx(pr,vas.mean(1),color=PUR,lw=1.8,marker='s',ms=3.4,ls='--',label='validation')
ax.fill_between(pr,vas.mean(1)-vas.std(1),vas.mean(1)+vas.std(1),color=PUR,alpha=0.15,lw=0)
kbest=int(np.argmax(vas.mean(1))); ax.plot(pr[kbest],vas.mean(1)[kbest],'*',color=GREEN,ms=12,mec='white',zorder=6)
NUM['val_curve']={'alpha_best':float(pr[kbest]),'val_best':round(float(vas.mean(1)[kbest]),4),'val_at_1':round(float(vas.mean(1)[-1]),4),'train_at_1':round(float(trs.mean(1)[-1]),4),'val_at_1e-7':round(float(vas.mean(1)[0]),4)}
ax.set_xlabel('strength  $\\alpha$'); ax.set_ylabel('accuracy'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8.5,loc='lower left',borderaxespad=0.3); ax.set_title('5-fold validation curve',fontsize=9.5,color=NAVY); ax.set_ylim(0.82,1.0)
ax=axs[1]; alpha=1e-3; s_=2*ytr-1; E_=np.arange(1,41)
def objective(m): f=Xs@m.coef_[0]+m.intercept_[0]; return np.maximum(0,1-s_*f).mean()+alpha/2*np.sum(m.coef_**2)
cfg={'constant, $\\eta_0=0.1$':dict(learning_rate='constant',eta0=0.1,tol=None,c=(BLUE,'-')),'adaptive, $\\eta_0=0.1$':dict(learning_rate='adaptive',eta0=0.1,n_iter_no_change=3,tol=1e-3,c=(RED,'-.')),
     'invscaling, $\\eta_0=0.1$':dict(learning_rate='invscaling',eta0=0.1,power_t=0.5,tol=None,c=(OR,'--')),'optimal':dict(learning_rate='optimal',tol=None,c=(GREEN,'-'))}
NUM['schedules']={}
for name,kw in cfg.items():
    kw=dict(kw); c_,ls_=kw.pop('c'); curve=[]
    for e in E_:
        o=[objective(SGDClassifier(alpha=alpha,random_state=sd,max_iter=int(e),**kw).fit(Xs,ytr)) for sd in range(5)]; curve.append(np.mean(o))
    ax.loglog(E_,curve,color=c_,ls=ls_,lw=1.8,label=name); NUM['schedules'][name]={'epoch1':round(curve[0],4),'epoch10':round(curve[9],4),'epoch40':round(curve[-1],4)}
ax.set_xlabel('epochs'); ax.set_ylabel('training objective'); ax.grid(color=GRID,lw=0.5,which='both'); ax.set_title('four schedules, 5 seeds',fontsize=9.5,color=NAVY); ax.legend(fontsize=7.8,loc='upper right',handlelength=2.0,borderaxespad=0.3,labelspacing=0.25)
ax.set_ylim(0.025,4)
fig.tight_layout(w_pad=1.0); save(fig,'f1821.png'); print(NUM['val_curve'],NUM['schedules']); json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

# ---- 17.13-1 case study figure: raw against scaled features
import re
rows=[]
for ln in open('verify/case_cancer.out').read().strip().split('\n')[1:]:
    m=re.match(r'(.+?)\s{2,}(raw|scaled)\s+([\d.]+)\s+([\d.]+) \+- ([\d.]+)',ln); rows.append((m.group(1).strip(),m.group(2),float(m.group(3)),float(m.group(4)),float(m.group(5))))
names=[]; [names.append(r[0]) for r in rows if r[0] not in names]
fig,axs=plt.subplots(1,2,figsize=(5.4,2.9),dpi=240,gridspec_kw={'width_ratios':[1.15,1]}); ax=axs[0]
for k,nm in enumerate(names[::-1]):
    for r in rows:
        if r[0]==nm:
            sc_=r[1]=='scaled'; yy=k+(0.12 if sc_ else -0.12)
            ax.errorbar(r[3],yy,xerr=r[4],fmt='o' if sc_ else 's',ms=4.5,color=BLUE if sc_ else OR,mfc=BLUE if sc_ else 'white',mec=BLUE if sc_ else OR,elinewidth=1.0,capsize=2,zorder=4)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names[::-1],fontsize=8.6); ax.set_xlabel('5-fold CV accuracy (mean $\\pm$ s.d.)'); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_xlim(0.6,1.02); ax.set_ylim(-0.6,len(names)-0.4)
ax.legend(handles=[Line2D([],[],marker='o',color=BLUE,ls='',ms=5,label='scaled'),Line2D([],[],marker='s',color=OR,mfc='white',ls='',ms=5,label='raw')],fontsize=8.5,loc='lower left',borderaxespad=0.3,handletextpad=0.2)
ax=axs[1]; ep=np.arange(1,31); sd_=range(5); NUM['case']={}
for tag,(A_,B_),c_,ls_ in [('scaled',(Xs,Xts),BLUE,'-'),('raw',(Xtr,Xte),OR,'--')]:
    acc=np.zeros((len(sd_),len(ep)))
    for j,sd in enumerate(sd_):
        m=SGDClassifier(random_state=sd)
        for k in range(len(ep)): m.partial_fit(A_,ytr,classes=[0,1]); acc[j,k]=m.score(B_,yte)
    ax.plot(ep,acc.mean(0),color=c_,ls=ls_,lw=1.8,label=tag); ax.fill_between(ep,acc.min(0),acc.max(0),color=c_,alpha=0.15,lw=0)
    NUM['case'][tag]={'epoch1':round(float(acc.mean(0)[0]),4),'epoch30':round(float(acc.mean(0)[-1]),4),'min':round(float(acc.min()),4),'max':round(float(acc.max()),4)}
ax.set_xlabel('epochs (calls of partial_fit)'); ax.set_ylabel('test accuracy'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8.5,loc='lower right',borderaxespad=0.3); ax.set_ylim(0.6,1.0)
ax.set_title('SGD hinge, 5 seeds',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=0.8); save(fig,'f1911.png')
# tuned alpha, evaluated once on the test set
mt=make_pipeline(StandardScaler(),SGDClassifier(alpha=NUM['val_curve']['alpha_best'],random_state=0)).fit(Xtr,ytr); NUM['case']['tuned']={'alpha':NUM['val_curve']['alpha_best'],'test':round(float(mt.score(Xte,yte)),4)}
print(NUM['case']); json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

# ---- 17.12-1 mistake bound against actual mistakes
from sklearn.svm import SVC
rng=np.random.default_rng(4); bounds=[]; mist=[]; gam=[]
for k in range(60):
    d=np.exp(rng.uniform(np.log(0.01),np.log(0.4))); th=rng.uniform(0,np.pi); w0=np.array([np.cos(th),np.sin(th)]); b0=rng.uniform(-0.3,0.3)
    P=rng.uniform(-1,1,(4000,2)); f0=P@w0+b0; P=P[np.abs(f0)>=d][:60]; yr=np.sign(P@w0+b0); Xr=P
    if len(yr)<60 or len(set(yr))<2: continue
    sv=SVC(kernel='linear',C=1e8).fit(Xr,yr); ustar=np.r_[sv.coef_[0],sv.intercept_[0]]; Xa=np.c_[Xr,np.ones(len(yr))]
    g=(yr*(Xa@ustar)).min()/np.linalg.norm(ustar); Rr=np.linalg.norm(Xa,axis=1).max()
    if g<=0: continue
    _,_,h=perceptron_run(Xr,yr,epochs=100000); mist.append(len(h)); bounds.append((Rr/g)**2); gam.append(g)
mist=np.array(mist); bounds=np.array(bounds); NUM['bound']={'n_sets':len(mist),'max_ratio':round(float((mist/bounds).max()),3),'median_ratio':round(float(np.median(mist/bounds)),3),'min_mist':int(mist.min()),'max_mist':int(mist.max())}
fig,ax=plt.subplots(figsize=(3.9,2.5),dpi=240)
ax.loglog([1,1e4],[1,1e4],color=GREEN,lw=1.5,label='mistakes = bound'); ax.scatter(bounds,mist,s=16,color=BLUE,edgecolor='white',lw=0.5,zorder=5,label='60 random data sets' if False else f'{len(mist)} random data sets')
ax.set_xlabel('bound  $R^2/\\gamma^2$'); ax.set_ylabel('mistakes made'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8.5,loc='upper left',borderaxespad=0.3); ax.set_xlim(10,1e4); ax.set_ylim(1,1e4)
plt.close(fig); print(NUM['bound']); json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)

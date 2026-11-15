# All figures of Chapter 1. Run: python3 fig/figs.py   (PNGs into ch01/, numbers into verify/fig_numbers.json)
import sys,os,json,runpy; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Rectangle
from sklearn.datasets import make_regression, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
plt.rcParams.update({'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8})
NUM={}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ---- 1.0-1 the learning loop
fig,ax=new(4.7,2.1)
def bx(xc,yc,w,h,t,fc,ec,fs=9.2,bold=True): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,bold=bold,r=0.08)
bx(1.05,2.0,1.9,0.95,'Data\n$\\mathbf{X}$ and $\\mathbf{y}$',LB,BLUE)
bx(4.6,2.0,2.5,0.95,'Model\n$\\hat y = f(\\mathbf{x};\\,\\mathbf{w}, b)$','#E3F4EA',GREEN)
bx(8.3,2.0,2.1,0.95,'Loss  $J$\nhow wrong?',PINK,RED)
bx(8.3,0.35,2.1,0.95,'Optimiser\nlower $J$',  '#FDF0E0',OR)
bx(1.05,0.35,1.9,0.95,'Prediction\n$\\hat y$ for new $\\mathbf{x}$',LGREY,GREY)
A=lambda p,q: arr(ax,p,q,c=NAVY,lw=1.1,ms=9)
A((2.0,2.0),(3.35,2.0)); ax.text(2.68,2.14,'$\\mathbf{x}$',fontsize=9,color=NAVY,ha='center')
A((5.85,2.0),(7.25,2.0)); ax.text(6.55,2.14,'$\\hat{\\mathbf{y}}$',fontsize=9,color=NAVY,ha='center')
ax.plot([1.05,1.05],[2.5,3.05],color=NAVY,lw=1.1); ax.plot([1.05,8.3],[3.05,3.05],color=NAVY,lw=1.1); A((8.3,3.05),(8.3,2.5)); ax.text(4.6,3.17,'known targets $\\mathbf{y}$',fontsize=9,color=NAVY,ha='center')
A((8.3,1.5),(8.3,0.83)); ax.text(8.45,1.15,'$\\nabla J$',fontsize=9,color=NAVY,ha='left',va='center')
ax.plot([7.25,5.2],[0.35,0.35],color=NAVY,lw=1.1); A((5.2,0.35),(5.2,1.5)); ax.text(6.25,0.5,'better $\\mathbf{w}$, $b$',fontsize=9,color=NAVY,ha='center')
ax.plot([3.95,3.95],[1.5,0.35],color=NAVY,lw=1.1); A((3.95,0.35),(2.0,0.35))
ax.text(2.97,0.45,'use when\n$J$ is small',fontsize=8.5,color=NAVY,ha='center',va='bottom')
ax.set_xlim(0,9.6); ax.set_ylim(-0.25,3.4); save(fig,'f100.png')

# ---- 1.1-1 a table becomes X and y
fig,ax=new(4.5,2.2)
flats=[(4,1,95),(5,2,120),(6,2,130),(7,3,160),(8,3,170),(9,4,200)]
cw=[0.55,0.85,0.8,0.85]; ch_=0.3; x0,y0=0.1,2.45
hd=['flat','area','rooms','price']; cols=[GREY,BLUE,BLUE,GREEN]; fcs=[LGREY,LB,LB,'#E3F4EA']
xs=np.cumsum([0]+cw)
for j in range(4):
    ax.add_patch(Rectangle((x0+xs[j],y0),cw[j],ch_,fc=cols[j],ec='white',lw=0.8)); ax.text(x0+xs[j]+cw[j]/2,y0+ch_/2,hd[j],ha='center',va='center',fontsize=8.5,color='white',weight='bold')
for i,row in enumerate(flats):
    for j,v in enumerate([i+1]+list(row)):
        ax.add_patch(Rectangle((x0+xs[j],y0-(i+1)*ch_),cw[j],ch_,fc=fcs[j],ec=cols[j],lw=0.6)); ax.text(x0+xs[j]+cw[j]/2,y0-(i+.5)*ch_,str(v),ha='center',va='center',fontsize=8.5,color=NAVY)
ax.text(x0+xs[1]+(cw[1]+cw[2])/2,y0+ch_+0.12,'features',ha='center',fontsize=9,color=BLUE,weight='bold'); ax.text(x0+xs[3]+cw[3]/2,y0+ch_+0.12,'target',ha='center',fontsize=9,color=GREEN,weight='bold')
ax.text(x0+xs[0]+cw[0]/2,y0+ch_+0.12,'sample',ha='center',fontsize=9,color=GREY,weight='bold')
ax.text(x0+xs[-1]/2,y0-6*ch_-0.12,'area in 10 m$^2$, price in thousands',ha='center',va='top',fontsize=8,color=GREY)
arr(ax,(3.45,1.55),(4.15,1.55),c=NAVY,lw=1.4,ms=12)
cx,cy=0.5,0.3
# X
X0=4.2
ax.text(X0+0.5,y0+ch_+0.12,'$\\mathbf{X}$',ha='center',fontsize=12,color=BLUE,weight='bold')
for i,row in enumerate(flats):
    for j in range(2):
        ax.add_patch(Rectangle((X0+j*0.5,y0-(i+1)*ch_),0.5,ch_,fc=LB,ec=BLUE,lw=0.6)); ax.text(X0+j*0.5+0.25,y0-(i+.5)*ch_,str(row[j]),ha='center',va='center',fontsize=8.5,color=NAVY)
Y0=5.55
ax.text(Y0+0.25,y0+ch_+0.12,'$\\mathbf{y}$',ha='center',fontsize=12,color=GREEN,weight='bold')
for i,row in enumerate(flats):
    ax.add_patch(Rectangle((Y0,y0-(i+1)*ch_),0.5,ch_,fc='#E3F4EA',ec=GREEN,lw=0.6)); ax.text(Y0+0.25,y0-(i+.5)*ch_,str(row[2]),ha='center',va='center',fontsize=8.5,color=NAVY)
ax.add_patch(Rectangle((X0,y0-3*ch_),1.0,ch_,fc='none',ec=OR,lw=1.7,zorder=6))
ax.text(X0-0.1,y0-6*ch_-0.12,'$\\mathbf{X}$: $n$ = 6 rows, $p$ = 2 columns\n$\\mathbf{y}$: $n$ = 6 values',ha='left',va='top',fontsize=8,color=GREY)
ax.text(X0-0.1,y0-6*ch_-0.85,'orange: row 3 is the sample $\\mathbf{x}_3$;\nits second entry is $x_{32}$ = 2',fontsize=8.2,color=OR,va='top',ha='left')
ax.set_xlim(0,6.5); ax.set_ylim(-0.75,3.05); save(fig,'f111.png')

# ---- 1.2-1 four kinds of task
fig,axs=plt.subplots(2,2,figsize=(4.4,3.7),dpi=240)
ax=axs[0,0]; Xr,yr=make_regression(n_samples=40,n_features=1,noise=14,random_state=2); m=LinearRegression().fit(Xr,yr); xx=np.array([[Xr.min()-0.1],[Xr.max()+0.1]])
ax.plot(xx,m.predict(xx),color=GREEN,lw=2); ax.scatter(Xr,yr,s=22,color=BLUE,edgecolor='white',lw=0.6,zorder=4); ax.set_title('Regression: predict a number',fontsize=9.5,color=NAVY); ax.set_xlabel('feature $x$'); ax.set_ylabel('target $y$ (a number)')
ax=axs[0,1]; Xc,yc=make_blobs(n_samples=60,centers=[(-1.5,-1),(1.5,1.2)],cluster_std=1.0,random_state=3); lr=LogisticRegression().fit(Xc,yc)
g1=np.linspace(-4.5,4.5,2); b0=lr.intercept_[0]; w0=lr.coef_[0]; ax.plot(g1,-(b0+w0[0]*g1)/w0[1],color=GREEN,lw=2)
ax.scatter(Xc[yc==0,0],Xc[yc==0,1],s=24,color=BLUE,edgecolor='white',lw=0.6,marker='o',zorder=4); ax.scatter(Xc[yc==1,0],Xc[yc==1,1],s=26,color=OR,edgecolor='white',lw=0.6,marker='s',zorder=4)
ax.set_xlim(-4.5,4.5); ax.set_ylim(-5.2,5.6); ax.set_title('Classification: predict a class',fontsize=9.5,color=NAVY); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$')
ax.text(-4.3,-4.9,'class 0 (circles)',color=BLUE,fontsize=9,va='bottom'); ax.text(4.3,5.3,'class 1 (squares)',color=OR,fontsize=9,ha='right',va='top')
ax=axs[1,0]; Xk,yk=make_blobs(n_samples=75,centers=[(-3,-1),(0,3),(3,-1)],cluster_std=0.85,random_state=4)
from matplotlib.patches import Ellipse
ax.scatter(Xk[:,0],Xk[:,1],s=22,color=GREY,edgecolor='white',lw=0.6,zorder=4)
for k in range(3):
    pk=Xk[yk==k]; ax.add_patch(Ellipse(pk.mean(0),4.4*pk[:,0].std(),4.4*pk[:,1].std(),fc='none',ec=GREEN,lw=1.6,ls=['-','--',':'][k],zorder=3))
ax.set_xlim(-6,6); ax.set_ylim(-3.2,5.4); ax.set_title('Clustering: find groups, no target',fontsize=9.5,color=NAVY); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$')
ax=axs[1,1]; rng=np.random.default_rng(6); t=rng.normal(0,2.2,45); Xd=np.c_[t,0.55*t]+rng.normal(0,0.45,(45,2)); pc=PCA(n_components=1).fit(Xd); z=pc.transform(Xd); Xp=pc.inverse_transform(z)
for a,b_ in zip(Xd,Xp): ax.plot([a[0],b_[0]],[a[1],b_[1]],color=GREY,lw=0.8,zorder=2)
d=pc.components_[0]*5.2; c0=pc.mean_; ax.plot([c0[0]-d[0],c0[0]+d[0]],[c0[1]-d[1],c0[1]+d[1]],color=GREEN,lw=2,zorder=3); ax.scatter(Xd[:,0],Xd[:,1],s=22,color=BLUE,edgecolor='white',lw=0.6,zorder=4)
ax.set_aspect('equal'); ax.set_xlim(-6.5,6.5); ax.set_ylim(-4.2,4.2); ax.set_title('Dimension reduction: 2 numbers to 1',fontsize=9.5,color=NAVY); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$')
NUM['pca_ratio']=round(float(PCA(n_components=1).fit(Xd).explained_variance_ratio_[0]),3)
for ax in axs.flat: grid(ax); ax.tick_params(labelsize=9)
fig.tight_layout(h_pad=1.0,w_pad=1.0); save(fig,'f121.png')

# ---- 1.3-1 knobs
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10])
fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; ax.axis('off'); ax.set_xlim(0,5); ax.set_ylim(0,4.5)
box(ax,0.1,1.5,0.9,0.8,'$x$',fc=LB,ec=BLUE,fs=11,r=0.08); box(ax,1.45,1.2,2.0,1.4,'$\\hat y = b + w\\,x$',fc='#E3F4EA',ec=GREEN,fs=10.5,r=0.08); box(ax,3.95,1.5,0.95,0.8,'$\\hat y$',fc=LGREY,ec=GREY,fs=11,r=0.08)
arr(ax,(1.0,1.9),(1.45,1.9),c=NAVY,lw=1.1,ms=9); arr(ax,(3.45,1.9),(3.95,1.9),c=NAVY,lw=1.1,ms=9)
for xc,lab in ((1.9,'$b$'),(3.0,'$w$')):
    ax.add_patch(Circle((xc,3.45),0.42,fc='#FDF0E0',ec=OR,lw=1.5)); ax.plot([xc,xc+0.2],[3.45,3.8],color=OR,lw=2); ax.text(xc,4.15,lab,ha='center',va='center',fontsize=10.5,color=OR)
ax.text(2.45,0.55,'two knobs, two numbers to learn',ha='center',fontsize=8.8,color=NAVY)
ax=axs[1]; xx=np.array([0,6.4]); 
for (b_,w_,c,ls,lab) in [(6.0,0.0,GREY,':','$w=0$: constant, $\\hat y = 6$'),(0.0,1.0,OR,'--','$b=0,\\ w=1$'),(0.9,1.7,GREEN,'-','$b=0.9,\\ w=1.7$')]:
    ax.plot(xx,b_+w_*xx,color=c,lw=1.8,ls=ls,label=lab)
ax.scatter(x,y,s=34,color=BLUE,edgecolor='white',lw=0.8,zorder=5); ax.set_xlim(0,6.4); ax.set_ylim(0,13.5); ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); grid(ax)
ax.legend(fontsize=8,loc='upper left',handlelength=2.4); ax.set_title('Three settings of the knobs',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f131.png')

# ---- 1.4-1 loss curves for the best constant
cn=runpy.run_path('code/constants.py'); yv=cn['y']; mse,mae=cn['mse'],cn['mae']
cc=np.linspace(0,20,400); fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240)
for ax,fn,cstar,nm,ttl in ((axs[0],mse,yv.mean(),'mean','Squared loss'),(axs[1],mae,np.median(yv),'median','Absolute loss')):
    ax.plot(cc,[fn(c) for c in cc],color=RED,lw=1.9); ax.plot(cstar,fn(cstar),'*',color=GREEN,ms=13,mec='white',zorder=6); ax.axvline(cstar,color=GREEN,lw=0.9,ls='--')
    ax.scatter(yv,np.zeros(5)-0.0,s=30,color=BLUE,edgecolor='white',lw=0.8,zorder=7,clip_on=False); ax.set_xlabel('constant prediction $c$ (min)'); ax.set_ylabel('loss $J(c)$'); grid(ax); ax.set_title(ttl,fontsize=9.5,color=NAVY); ax.set_xlim(0,20)
axs[0].set_ylim(0,150); axs[1].set_ylim(0,10)
axs[0].text(6.5,100,'minimum at the mean\n$c$ = 6, $J$ = 26',fontsize=8.5,color=GREEN); axs[1].text(8.2,1.9,'minimum at the median\n$c$ = 4, $J$ = 3.2',fontsize=8.5,color=GREEN)
axs[0].annotate('the point 16\npulls the mean',xy=(16,0.5),xytext=(11.5,45),fontsize=8.3,color=BLUE,arrowprops=dict(arrowstyle='-',color=BLUE,lw=0.6))
axs[1].text(6.3,0.55,'data (blue)',fontsize=8.3,color=BLUE)
fig.tight_layout(w_pad=1.2); save(fig,'f141.png')

# ---- 1.5-1 gradient descent paths
ds=runpy.run_path('code/descent.py'); desc=ds['descend']; s2=float(yv.var()); ybar=float(yv.mean()); Jc=lambda c:(c-ybar)**2+s2
fig,axs=plt.subplots(1,3,figsize=(4.8,2.0),dpi=240,sharey=True)
cx_=np.linspace(-9,22,300)
for ax,eta,ttl in zip(axs,[0.25,0.9,1.1],['$\\eta$ = 0.25','$\\eta$ = 0.9','$\\eta$ = 1.1']):
    ax.plot(cx_,Jc(cx_),color=RED,lw=1.5); p=desc(eta,steps=5); ax.plot(p,Jc(p),'-o',color=GREEN if eta<1 else PUR,ms=3.8,lw=1.0,zorder=5)
    ax.plot(p[0],Jc(p[0]),'s',color=NAVY,ms=5,zorder=6); ax.set_title(ttl,fontsize=8.8,color=NAVY); ax.set_xlabel('$c$ (min)'); ax.set_xlim(-9,22); ax.set_ylim(0,300); grid(ax); ax.tick_params(labelsize=8)
axs[0].set_ylabel('loss $J(c)$'); axs[0].text(0.3,85,'start $c_0$ = 0',fontsize=8,color=NAVY,ha='left',va='bottom')
fig.tight_layout(w_pad=0.5); save(fig,'f151.png')

# ---- 1.6-1 under- and over-fitting
gn=runpy.run_path('code/generalise.py'); xtr,ytr,xte,yte=gn['x_tr'],gn['y_tr'],gn['x_te'],gn['y_te']; P=gn['Polynomial']; rm=gn['rmse']
fig,axs=plt.subplots(1,4,figsize=(4.9,1.9),dpi=240,gridspec_kw={'width_ratios':[1,1,1,1.15]}); xg=np.linspace(0,1,300); tr_e=[];te_e=[];degs=[0,1,3,9]
for ax,dg,ttl in zip(axs[:3],[1,3,9],['degree 1','degree 3','degree 9']):
    f=P.fit(xtr,ytr,dg); ax.plot(xg,np.sin(2*np.pi*xg),color=GRID,lw=1.6,zorder=1); ax.plot(xg,f(xg),color=GREEN,lw=1.8,zorder=3)
    ax.scatter(xte,yte,s=9,color=GREY,alpha=0.55,lw=0,zorder=2); ax.scatter(xtr,ytr,s=22,color=BLUE,edgecolor='white',lw=0.6,zorder=5); ax.set_ylim(-1.8,1.8); ax.set_xlim(0,1); ax.set_title(ttl,fontsize=8.3,color=NAVY); ax.set_xlabel('$x$'); grid(ax); ax.tick_params(labelsize=7.5)
axs[0].set_ylabel('$y$')
for dg in degs: f=P.fit(xtr,ytr,dg); tr_e.append(rm(ytr,f(xtr))); te_e.append(rm(yte,f(xte)))
ax=axs[3]; ix=np.arange(4); ax.bar(ix-0.19,tr_e,0.36,color=BLUE,label='train'); ax.bar(ix+0.19,[min(v,1.6) for v in te_e],0.36,color=GREY,label='test',hatch='//',edgecolor='white')
ax.text(3.19,1.62,f'{te_e[3]:.1f}',ha='center',va='bottom',fontsize=7.5,color=GREY); ax.set_xticks(ix); ax.set_xticklabels([str(d) for d in degs],fontsize=7.5); ax.set_xlabel('degree'); ax.set_ylabel('RMSE'); ax.set_ylim(0,1.95); ax.legend(fontsize=7.5,loc='upper left',handlelength=1.2); ax.set_title('error',fontsize=8.3,color=NAVY); grid(ax); ax.tick_params(labelsize=7.5)
fig.tight_layout(w_pad=0.5); save(fig,'f161.png')

# ---- 1.8-1 a map of the series
fig,ax=new(5.0,2.95); ax.set_aspect('auto')
def cells(row,y0,c0,c1,col,x00=0.15,cwid=0.236):
    for k in range(c0,c1+1):
        xl=x00+(k-c0)*cwid; ax.add_patch(Rectangle((xl,y0),cwid-0.02,0.34,fc=col[k] if isinstance(col,dict) else col,ec='white',lw=0.6)); ax.text(xl+(cwid-0.02)/2,y0+0.17,str(k),ha='center',va='center',fontsize=7.6,color='white',weight='bold')
def brk(x1,x2,y,txt,col=NAVY,dy=-0.08,fs=8,ha='center'):
    ax.plot([x1,x1,x2,x2],[y+0.06,y,y,y+0.06],color=col,lw=0.9); ax.text((x1+x2)/2 if ha=='center' else x1,y-0.08,txt,ha=ha,va='top',fontsize=fs,color=col,linespacing=1.15)
PC={'A':BLUE,'B':TEAL,'C':GREEN,'D':OR}; cmap={}
for k in range(1,20): cmap[k]=PC['A' if k<=6 else 'B' if k<=8 else 'C' if k<=15 else 'D']
cw_=0.236
ax.text(0.15,3.4,'VOLUME I   Foundations and Linear Models   (this book)',fontsize=9.5,color=NAVY,weight='bold',va='bottom')
cells(0,3.05,1,19,cmap)
for (a,b_,lab,c) in ((1,6,'A\nToolkit',BLUE),(7,8,'\n\nB\nMachinery',TEAL),(9,15,'C\nLinear regression',GREEN),(16,19,'D\nLinear\nclassification',OR)):
    brk(0.15+(a-1)*cw_,0.15+b_*cw_-0.02,3.0,lab,c,fs=8.2)
ax.text(0.15,1.3,'VOLUME II   Chapters 20 to 35',fontsize=9.5,color=NAVY,weight='bold',va='bottom')
cells(1,0.9,20,35,GREY); brk(0.15,0.15+16*cw_-0.02,0.85,'neighbours, kernels, SVMs, trees, ensembles,\nGaussian processes, neural networks, calibration,\nmulticlass wrappers, semi-supervised learning, PLS',GREY,fs=7.8)
ax.text(0.15,-0.55,'VOLUME III   Chapters 36 to 59',fontsize=9.5,color=NAVY,weight='bold',va='bottom')
cells(2,-0.95,36,59,GREY)
brk(0.15,0.15+13*cw_-0.02,-1.0,'clustering, dimension reduction,\ncovariance, density, outliers',GREY,fs=8.0)
brk(0.15+13*cw_,0.15+18*cw_-0.02,-1.0,'preproc-\nessing',GREY,fs=7.8)
brk(0.15+18*cw_,0.15+24*cw_-0.02,-1.0,'pipelines, validation,\nsearch, metrics,\nthresholds, playbook',GREY,fs=8.0)
ax.set_xlim(0,5.95); ax.set_ylim(-1.95,3.7); save(fig,'f181.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM)); print('figs ok')

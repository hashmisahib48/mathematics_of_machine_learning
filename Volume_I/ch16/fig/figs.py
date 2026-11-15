# All figures of Chapter 16. Run: python3 fig/figs.py  (PNGs into ch16/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
OFF=LinearSegmentedColormap.from_list('bo',['#BBD3F2','#FFFFFF','#FBD9A8'])   # class 0 (blue) -> class 1 (orange)
GR='#E3F4EA'
hours=np.arange(1.,9.); passed=np.array([0,0,0,1,0,1,1,1.]); XA=np.c_[np.ones(8),hours]
sig=lambda z:1/(1+np.exp(-z))
def newton(X,y,steps=12,w0=None):
    w=np.zeros(X.shape[1]) if w0 is None else w0.copy(); path=[w.copy()]
    for _ in range(steps):
        p=sig(X@w); g=X.T@(p-y); H=X.T@(X*(p*(1-p))[:,None]); w=w-np.linalg.solve(H,g); path.append(w.copy())
    return w,np.array(path)
wopt,path=newton(XA,passed); B0,W0=wopt
NUM['toy']={'b':round(float(B0),4),'w':round(float(W0),4),'boundary':round(float(-B0/W0),4)}
def logloss(X,y,w):
    z=X@w; return float(np.sum(np.logaddexp(0,z)-y*z))
NUM['toy']['J_opt']=round(logloss(XA,passed,wopt),4)
def pts01(ax,x=hours,y=passed,s=34,jit=0):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(x[m],y[m],s=s,color=col,marker=mk,edgecolor='white',lw=0.8,zorder=5)

# ---- 16.0-1 opener: the S-curve through pass/fail
fig,ax=plt.subplots(figsize=(4.2,2.15),dpi=240)
xx=np.linspace(0,9,300); ax.plot(xx,sig(B0+W0*xx),color=GREEN,lw=2.2,zorder=3)
ax.axhline(0.5,color=GRID,lw=0.8,zorder=1); ax.axvline(-B0/W0,color=PUR,lw=1,ls='--',zorder=1)
for xi,yi in zip(hours,passed): ax.plot([xi,xi],[yi,sig(B0+W0*xi)],color=RED,lw=0.9,alpha=0.7,zorder=2)
pts01(ax)
ax.text(-B0/W0+0.12,0.06,'boundary\n4.5 hours',color=PUR,fontsize=9,va='bottom')
ax.text(8.9,0.17,'predicted\nprobability\nof passing',color=GREEN,fontsize=9,ha='right',va='bottom',weight='bold')
ax.text(0.1,1.04,'passed',color=OR,fontsize=9,va='center'); ax.text(0.1,0.13,'failed',color=BLUE,fontsize=9,va='center')
ax.set_xlim(0,9); ax.set_ylim(-0.13,1.13); ax.set_yticks([0,0.5,1]); ax.set_xlabel('hours of study  $x$'); ax.set_ylabel('$P$(pass)')
sv(fig,'f160.png',400)

# ---- 16.1-1 why not a straight line
m_=np.polyfit(hours,passed,1); fig,ax=plt.subplots(figsize=(3.8,2.3),dpi=240)
xx=np.linspace(-1,10,200); ax.plot(xx,np.polyval(m_,xx),color=GREY,lw=1.5,ls='--'); ax.plot(xx,sig(B0+W0*xx),color=GREEN,lw=2)
ax.axhspan(-0.5,0,color=PINK,zorder=0); ax.axhspan(1,1.5,color=PINK,zorder=0); pts01(ax)
ax.text(9.9,1.25,'above 1: not a probability',color=RED,fontsize=8.5,ha='right',va='center'); ax.text(9.9,-0.25,'below 0',color=RED,fontsize=8.5,ha='right',va='center')
ax.text(-0.7,0.78,'least-squares\nline',color=GREY,fontsize=8.5,ha='left'); ax.text(6.6,0.52,'logistic\ncurve',color=GREEN,fontsize=8.5,ha='left',weight='bold')
ax.set_xlim(-1,10); ax.set_ylim(-0.5,1.5); ax.set_yticks([0,0.5,1]); ax.set_xlabel('hours of study  $x$'); ax.set_ylabel('pass (1) or fail (0)')
NUM['ols_line']={'at0':round(float(np.polyval(m_,0)),3),'at10':round(float(np.polyval(m_,10)),3),'slope':round(float(m_[0]),4),'icpt':round(float(m_[1]),4)}
sv(fig,'f161.png',360)

# ---- 16.2-1 odds, log-odds, sigmoid
fig,axs=plt.subplots(1,2,figsize=(4.9,2.5),dpi=240); ax=axs[0]; z=np.linspace(-6,6,300); ax.plot(z,sig(z),color=GREEN,lw=2)
for zz in (-2.2,0,1.1): ax.plot(zz,sig(zz),'o',color=NAVY,ms=4.5,zorder=5); ax.plot([zz,zz],[0,sig(zz)],color=GRID,lw=0.8,zorder=1)
ax.text(1.3,0.70,'    $z$        $p$       odds\n$-$2.2    0.10     1 : 9\n  0.0     0.50     1 : 1\n  1.1     0.75     3 : 1',fontsize=8.2,color=NAVY,va='top',ha='left',linespacing=1.35,bbox=dict(fc='white',ec=GRID,pad=2.0,lw=0.6))
ax.set_xlabel('log-odds  $z=b+wx$'); ax.set_ylabel('probability  $p=\\sigma(z)$'); ax.set_xlim(-6,6); ax.set_ylim(-0.02,1.02); ax.set_title('log-odds to probability',fontsize=9.5,color=NAVY)
ax=axs[1]; xx=np.linspace(0,9,300)
for wv,b_,ls,c,lab,lw in ((0.45,-0.45*4.5,':',GREY,'$w$ = 0.45',1.6),(W0,B0,'-',GREEN,'$w$ = 1.28',2.2),(3.0,-3*4.5,'--',GREY,'$w$ = 3',1.4),(W0,B0+3,'-.',OR,'$b$ + 3, $w$ = 1.28',1.7)):
    ax.plot(xx,sig(b_+wv*xx),color=c,ls=ls,lw=lw,label=lab)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.3),ncol=2,fontsize=8.5,handlelength=2.2,columnspacing=1.2)
ax.set_xlabel('hours  $x$'); ax.set_ylabel('$P$(pass)'); ax.set_xlim(0,9); ax.set_ylim(-0.02,1.02); ax.set_title('what the two numbers do',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.2); sv(fig,'f162.png',470)

# ---- 16.3-1 log loss curves and per-student loss
fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240,gridspec_kw={'width_ratios':[1,1.3]}); ax=axs[0]; pp=np.linspace(0.005,0.995,300)
ax.plot(pp,-np.log(pp),color=OR,lw=2,label='true class 1: $-\\ln p$'); ax.plot(pp,-np.log(1-pp),color=BLUE,lw=2,ls='--',label='true class 0: $-\\ln(1-p)$')
ax.legend(loc='upper center',fontsize=8.2,handlelength=1.8,borderaxespad=0.1)
ax.set_xlabel('predicted probability  $p$'); ax.set_ylabel('loss  $\\ell$'); ax.set_ylim(0,4.3); ax.set_xlim(0,1); ax.grid(color=GRID,lw=0.5); ax.set_title('cost of one prediction',fontsize=9.5,color=NAVY)
ax=axs[1]; pf=sig(XA@wopt); li=-(passed*np.log(pf)+(1-passed)*np.log(1-pf)); NUM['toy']['loss_each']=[round(float(v),4) for v in li]
ax.bar(hours,li,color=[OR if v else BLUE for v in passed],width=0.62,edgecolor='white',hatch=None)
for xi,v,c in zip(hours,li,passed): ax.text(xi,v+0.04,f'{v:.2f}',ha='center',fontsize=8,color=NAVY)
ax.set_xlabel('student  (hours studied)'); ax.set_ylabel('loss  $\\ell_i$'); ax.set_ylim(0,1.35); ax.set_xticks(hours); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.set_title(f'at the fitted model: $J$ = {li.sum():.3f}',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.2); sv(fig,'f163.png',450)

# ---- 16.3-2 two features: probability surface and distance
Xs,ys=make_classification(n_samples=70,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,class_sep=1.1,flip_y=0.04,random_state=11)
lr2=LogisticRegression(C=10).fit(Xs,ys); w2=lr2.coef_[0]; b2=lr2.intercept_[0]; nw=np.linalg.norm(w2)
NUM['two']={'w':[round(float(v),3) for v in w2],'b':round(float(b2),3),'norm':round(float(nw),3),'acc':round(float(lr2.score(Xs,ys)),4)}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.35),dpi=240,gridspec_kw={'width_ratios':[1.15,1]}); ax=axs[0]
gx,gy=np.meshgrid(np.linspace(Xs[:,0].min()-0.8,Xs[:,0].max()+0.8,200),np.linspace(Xs[:,1].min()-0.8,Xs[:,1].max()+0.8,200))
P=lr2.predict_proba(np.c_[gx.ravel(),gy.ravel()])[:,1].reshape(gx.shape)
ax.contourf(gx,gy,P,levels=np.linspace(0,1,11),cmap=OFF,zorder=0)
ax.contour(gx,gy,P,levels=[0.1,0.25,0.75,0.9],colors=GREY,linewidths=0.6,linestyles=':',zorder=1); ax.contour(gx,gy,P,levels=[0.5],colors=GREEN,linewidths=2,zorder=2)
for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')): m=ys==cls; ax.scatter(Xs[m,0],Xs[m,1],s=18,color=col,marker=mk,edgecolor='white',lw=0.6,zorder=5)
c0=np.array([0.5,-1.0]); c0=c0-(b2+w2@c0)/nw**2*w2; arr(ax,tuple(c0),tuple(c0+0.9*w2/nw),c=PUR,lw=1.6,ms=9)
ax.text(c0[0]+0.9*w2[0]/nw+0.12,c0[1]+0.9*w2[1]/nw-0.05,'$\\mathbf{w}$',color=PUR,fontsize=10,va='top')
ax.text(0.03,0.97,'$p$ = 0.5: green\ndotted: 0.1, 0.25,\n0.75, 0.9',transform=ax.transAxes,fontsize=8,color=NAVY,va='top',bbox=dict(fc='white',ec='none',pad=1.5,alpha=0.85))
ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$'); ax.set_xlim(gx.min(),gx.max()); ax.set_ylim(gy.min(),gy.max()); ax.set_title('probability of class 1',fontsize=9.5,color=NAVY)
ax=axs[1]; dd=np.linspace(-6,6,200); d_=(Xs@w2+b2)/nw; ax.plot(dd,sig(nw*dd),color=GREEN,lw=2)
for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')): m=ys==cls; ax.scatter(d_[m],sig(nw*d_[m]),s=14,color=col,marker=mk,edgecolor='white',lw=0.5,zorder=5)
ax.axvline(0,color=GRID,lw=0.8); ax.set_xlim(-5,5); ax.set_xlabel('signed distance  $d$  from the line'); ax.set_ylabel('$P$(class 1)'); ax.set_title('depends on $d$ alone',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig,'f164.png',470)

# ---- 16.3-3 Newton path and convergence against gradient descent
bg,wg=np.meshgrid(np.linspace(-8,1.5,260),np.linspace(-0.3,2.2,260)); Jg=np.array([[logloss(XA,passed,np.array([b_,w_])) for b_ in bg[0]] for w_ in wg[:,0]])
fig,axs=plt.subplots(1,2,figsize=(4.9,2.3),dpi=240); ax=axs[0]
cs=ax.contourf(bg,wg,Jg,levels=[2.5,3,4,5,6,8,12,16,24,40],cmap='Blues_r',alpha=0.9); ax.contour(bg,wg,Jg,levels=[3,4,5,6,8,12,16,24],colors='white',linewidths=0.5)
ax.plot(path[:7,0],path[:7,1],'-o',color=GREEN,ms=4,lw=1.5,zorder=5,mec='white',mew=0.6)
for k in range(4): ax.text(path[k,0]-0.4,path[k,1]-0.2,f'{k}',fontsize=8.5,color=NAVY,weight='bold',bbox=dict(fc='white',ec='none',pad=0.8,alpha=0.85),ha='right',va='center')
ax.plot(wopt[0],wopt[1],'*',color=OR,ms=11,mec='white',zorder=6)
ax.set_xlabel('intercept $b$'); ax.set_ylabel('weight $w$'); ax.set_title('Newton steps 0, 1, 2, 3 ...',fontsize=9.5,color=NAVY); ax.set_xlim(-8,1.5); ax.set_ylim(-0.3,2.2)
ax=axs[1]; Jopt=logloss(XA,passed,wopt); itN=range(0,8); errN=[max(logloss(XA,passed,path[k])-Jopt,1e-14) for k in itN]
v=np.zeros(2); errG=[]; eta=0.03
for k in range(31):
    errG.append(max(logloss(XA,passed,v)-Jopt,1e-14)); p_=sig(XA@v); v=v-eta*XA.T@(p_-passed)
ax.semilogy(list(itN),errN,'-o',color=GREEN,ms=3.5,lw=1.6,label='Newton'); ax.semilogy(range(31),errG,'--s',color=GREY,ms=2.5,lw=1.0,markevery=3,label='gradient descent,\n$\\eta$ = 0.03')
ax.set_ylim(1e-14,1e1); ax.set_xlabel('iteration'); ax.set_ylabel('$J-J_{\\min}$'); ax.legend(loc='center right',fontsize=8.5); ax.grid(color=GRID,lw=0.5); ax.set_title('distance to the minimum',fontsize=9.5,color=NAVY)
NUM['newton']={'path':[[round(float(a),4),round(float(b),4)] for a,b in path[:7]],'gd_err_after_30':float(f'{errG[-1]:.3g}'),'newton_err':[float(f'{e:.2g}') for e in errN]}
fig.tight_layout(w_pad=1.0); sv(fig,'f165.png',470)

# ---- 16.4-1 three classes: regions and probabilities along a line
iris=load_iris(); Xi=iris.data[:,2:4]; yi=iris.target; sm=LogisticRegression(C=10,max_iter=2000).fit(Xi,yi)
NUM['iris2']={'acc':round(float(sm.score(Xi,yi)),4)}
CL=[BLUE,OR,TEAL]; MK=['o','s','^']; names=['setosa','versicolor','virginica']
fig,axs=plt.subplots(1,2,figsize=(4.9,2.35),dpi=240,gridspec_kw={'width_ratios':[1.1,1]}); ax=axs[0]
gx,gy=np.meshgrid(np.linspace(0.5,7.2,300),np.linspace(-0.1,2.9,300)); Z=sm.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5,2.5],colors=['#DCE8F8','#FBE7C9','#D4ECEE'],zorder=0); ax.contour(gx,gy,Z,levels=[0.5,1.5],colors=GREEN,linewidths=1.8,zorder=2)
for k in range(3): m=yi==k; ax.scatter(Xi[m,0]+0.0,Xi[m,1],s=13,color=CL[k],marker=MK[k],edgecolor='white',lw=0.4,zorder=5)
cent=[Xi[yi==k].mean(0) for k in (0,2)]; ax.plot([cent[0][0],cent[1][0]],[cent[0][1],cent[1][1]],color=PUR,lw=1,ls='--',zorder=3)
for k,(tx_,ty_) in enumerate([(1.0,1.3),(3.9,0.45),(5.2,2.65)]): ax.text(tx_,ty_,names[k],color=CL[k],fontsize=8.5,weight='bold',ha='left',va='center',bbox=dict(fc='white',ec='none',pad=0.8,alpha=0.8))
ax.set_xlabel('petal length (cm)'); ax.set_ylabel('petal width (cm)'); ax.set_title('three regions, two straight borders',fontsize=9,color=NAVY); ax.set_xlim(0.5,7.2); ax.set_ylim(-0.1,2.9)
ax=axs[1]; tt=np.linspace(-0.1,1.1,200); line=np.array(cent[0])[None]+tt[:,None]*(np.array(cent[1])-np.array(cent[0]))[None]; Pk=sm.predict_proba(line)
for k,ls in zip(range(3),['-','--','-.']): ax.plot(tt,Pk[:,k],color=CL[k],ls=ls,lw=1.8,label=names[k])
ax.legend(loc='center left',bbox_to_anchor=(0.0,0.45),fontsize=8.5,handlelength=2.4)
ax.set_xlabel('position on the dashed line'); ax.set_ylabel('probability $P_k$'); ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.02,1.02); ax.set_title('the three $P_k$ add up to 1',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig,'f166.png',470)

# ---- 16.7-1 flow chart of fit()
fig,ax=new(3.8,5.1)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3):
    ax.add_patch(FancyBboxPatch((xc-w/2,yc-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.15)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.4,7.6,6.4,0.55,'Input:  $\\mathbf{X}$,  labels $y$,  optional weights',fc=LGREY,ec=GREY)
fb(3.4,6.65,6.4,0.55,'Validate;  find  classes_  ($K$ labels)')
fb(3.4,5.5,6.4,1.05,'Choose the loss:\n$K=2$: log loss, one weight vector\n$K\\geq3$: softmax loss, $K$ weight vectors')
fb(3.4,4.25,6.4,1.0,'Weights: sample_weight $\\times$ class_weight\nBuild  $J$  of Eq. 16.5-1\nfrom  $C$  and  l1_ratio')
ax.add_patch(Polygon([(3.4,3.6),(4.8,3.15),(3.4,2.7),(2.0,3.15)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.4,3.15,'solver?',ha='center',va='center',fontsize=8.6,color=NAVY)
fb(1.15,1.85,2.2,1.2,'lbfgs (default)\nnewton-cg\nnewton-cholesky\nL2 or none',fc='#E3F4EA',ec=GREEN,fs=8)
fb(3.4,1.85,2.0,1.2,'liblinear\nbinary only\nL1 or L2',fc='#F1EAF7',ec=PUR,fs=8)
fb(5.65,1.85,2.2,1.2,'sag, saga\nsaga: L1 and\nelastic net',fc='#F1EAF7',ec=PUR,fs=8)
fb(3.4,0.55,6.4,0.7,'Iterate until the gradient is below tol\nor max_iter is reached',fc=PINK,ec=RED,fs=8.3)
fb(3.4,-0.5,6.4,0.55,'Store  coef_, intercept_, n_iter_;  return self',fc=LGREY,ec=GREY)
A((3.4,7.32),(3.4,6.93)); A((3.4,6.37),(3.4,6.03)); A((3.4,4.97),(3.4,4.77)); A((3.4,3.75),(3.4,3.61))
ax.plot([2.0,1.15],[3.15,3.15],color=NAVY,lw=0.9); A((1.15,3.15),(1.15,2.45)); A((3.4,2.7),(3.4,2.45)); ax.plot([4.8,5.65],[3.15,3.15],color=NAVY,lw=0.9); A((5.65,3.15),(5.65,2.45))
for xs in (1.15,3.4,5.65): ax.plot([xs,xs],[1.25,1.0],color=NAVY,lw=0.9)
ax.plot([1.15,5.65],[1.0,1.0],color=NAVY,lw=0.9); A((3.4,1.0),(3.4,0.9))
A((3.4,0.2),(3.4,-0.22))
ax.set_xlim(0,6.8); ax.set_ylim(-0.85,7.95); sv(fig,'f167.png',360)

# ---- 16.10-1 C gallery
Xg,yg=make_classification(n_samples=90,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,class_sep=0.9,flip_y=0.06,random_state=4)
Xg=StandardScaler().fit_transform(Xg); NUM['gallery']={}
fig,axs=plt.subplots(2,2,figsize=(4.9,4.5),dpi=240,sharex=True,sharey=True)
gx,gy=np.meshgrid(np.linspace(-3,3,220),np.linspace(-3,3,220))
for ax,C in zip(axs.ravel(),(0.001,0.03,1,1000)):
    m=LogisticRegression(C=C,tol=1e-10,max_iter=5000).fit(Xg,yg); P=m.predict_proba(np.c_[gx.ravel(),gy.ravel()])[:,1].reshape(gx.shape)
    ax.contourf(gx,gy,P,levels=np.linspace(0,1,11),cmap=OFF,zorder=0); ax.contour(gx,gy,P,levels=[0.1,0.9],colors=GREY,linewidths=0.7,linestyles=':'); ax.contour(gx,gy,P,levels=[0.5],colors=GREEN,linewidths=2)
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')): k=yg==cls; ax.scatter(Xg[k,0],Xg[k,1],s=13,color=col,marker=mk,edgecolor='white',lw=0.5,zorder=5)
    nw_=float(np.linalg.norm(m.coef_)); NUM['gallery'][str(C)]={'norm':round(nw_,3),'acc':round(float(m.score(Xg,yg)),4),'maxp':round(float(m.predict_proba(Xg)[:,1].max()),3)}
    ax.set_title(f'$C$ = {C:g}',fontsize=10,color=PUR,weight='bold'); ax.set_xlim(-3,3); ax.set_ylim(-3,3)
    ax.text(0.04,0.04,f'$\\Vert\\mathbf{{w}}\\Vert$ = {nw_:.2f}\naccuracy {m.score(Xg,yg):.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,va='bottom',bbox=dict(fc='white',ec='none',pad=1.5,alpha=0.85))
for ax in axs[1]: ax.set_xlabel('feature $x_1$')
for ax in axs[:,0]: ax.set_ylabel('feature $x_2$')
fig.tight_layout(h_pad=0.8,w_pad=0.6); sv(fig,'f168.png',470)

# ---- breast-cancer data, shared by the next figures
bc=load_breast_cancer(); Xb_tr,Xb_te,yb_tr,yb_te=train_test_split(bc.data,bc.target,test_size=0.25,random_state=0,stratify=bc.target)
sc=StandardScaler().fit(Xb_tr); Zb=sc.transform(Xb_tr)
# ---- 16.10-2 validation curve
Cs=np.logspace(-4,2,13); pipe=make_pipeline(StandardScaler(),LogisticRegression())
tr,va=validation_curve(pipe,Xb_tr,yb_tr,cv=5,scoring='neg_log_loss',param_name='logisticregression__C',param_range=Cs)
fig,ax=plt.subplots(figsize=(4.0,2.35),dpi=240); best=Cs[np.argmax(va.mean(1))]
ax.semilogx(Cs,-tr.mean(1),'-o',color=BLUE,ms=3.5,lw=1.6,label='training'); ax.semilogx(Cs,-va.mean(1),'--s',color=GREY,ms=3.5,lw=1.6,label='validation (5 folds)')
ax.fill_between(Cs,-va.mean(1)-va.std(1),-va.mean(1)+va.std(1),color=GREY,alpha=0.15,lw=0)
ax.axvline(best,color=PUR,lw=1.2,ls=':'); ax.text(best*1.25,0.27,f'best $C$ = {best:.2g}',color=PUR,fontsize=8.5,va='center')
ax.text(2e-4,0.06,'too much penalty:\nunderfitting',fontsize=8.5,color=NAVY,va='bottom'); ax.text(100,0.69,'too little:\noverfitting',fontsize=8.5,color=NAVY,ha='right',va='top')
ax.set_xlabel('$C$  (weaker penalty to the right)'); ax.set_ylabel('mean log loss'); ax.set_ylim(0,0.72); ax.grid(color=GRID,lw=0.5); ax.legend(loc='upper center',bbox_to_anchor=(0.36,1.0),fontsize=8.5,ncol=1,borderaxespad=0.1)
NUM['valcurve']={'best':float(best),'val_at_best':round(float(-va.mean(1).max()),4)}
sv(fig,'f169.png',380)

# ---- 16.10-3 coefficient paths, L2 and L1
Cp=np.logspace(-3,1.5,22); P2=[];P1=[];NZ=[]
for C in Cp:
    P2.append(LogisticRegression(C=C,max_iter=5000).fit(Zb,yb_tr).coef_[0])
    m1=LogisticRegression(C=C,l1_ratio=1.0,solver='liblinear',max_iter=5000,tol=1e-6).fit(Zb,yb_tr); P1.append(m1.coef_[0]); NZ.append(int((m1.coef_!=0).sum()))
P2=np.array(P2);P1=np.array(P1); w1=LogisticRegression(max_iter=5000).fit(Zb,yb_tr).coef_[0]; top=np.argsort(-abs(w1))[:3]
sty=[(RED,'-'),(PUR,'--'),(GREEN,'-.')]
fig,axs=plt.subplots(1,2,figsize=(4.9,2.4),dpi=240,sharey=True)
for ax,Pm,t in zip(axs,(P2,P1),('L2 penalty','L1 penalty (boxes: nonzero)')):
    for j in range(30): ax.semilogx(Cp,Pm[:,j],color='#C8CDD5',lw=0.7)
    for (c_,ls),j in zip(sty,top): ax.semilogx(Cp,Pm[:,j],color=c_,ls=ls,lw=1.7)
    ax.set_title(t,fontsize=9.5,color=NAVY); ax.set_xlabel('$C$'); ax.grid(color=GRID,lw=0.5); ax.axhline(0,color=GREY,lw=0.6)
axs[0].set_ylabel('weight  $w_j$')
for k,(c_,ls) in zip(range(3),sty): axs[0].text(0.0012,[4.5,3.7,2.9][k],bc.feature_names[top[k]],color=c_,fontsize=8.5,ha='left',va='center',weight='bold')
NUM['paths_boxes']={}
for Cq in (0.02,0.1,0.5): k=int(np.argmin(abs(Cp-Cq))); NUM['paths_boxes'][f'{Cp[k]:.3g}']=NZ[k]; axs[1].text(Cp[k],-3.6,f'{NZ[k]}',fontsize=8.5,color=NAVY,ha='center',bbox=dict(fc='white',ec=GRID,pad=1.2,lw=0.5))
NUM['paths']={'nonzero':{f'{Cp[k]:.3g}':NZ[k] for k in range(0,22,3)},'top':[bc.feature_names[j] for j in top]}
axs[0].set_ylim(-4.3,5.2); fig.tight_layout(w_pad=0.8); sv(fig,'f1610.png',470)

# ---- 16.11-1 separable data
ysep=np.array([0,0,0,1,1,1,1,1.]); fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240); ax=axs[0]; xx=np.linspace(0,9,300)
for s_,ls,lw in ((1,':',1.5),(4,'--',1.5),(16,'-',2)): ax.plot(xx,sig(s_*(xx-3.5)),color=GREEN,ls=ls,lw=lw)
for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')): m=ysep==cls; ax.scatter(hours[m],ysep[m],s=30,color=col,marker=mk,edgecolor='white',lw=0.7,zorder=5)
ax.text(5.0,0.2,'$w$ = 1, 4, 16\n(same boundary)',fontsize=8.5,color=GREEN,va='bottom'); ax.set_xlabel('hours  $x$'); ax.set_ylabel('$P$(pass)'); ax.set_title('three ever sharper curves',fontsize=9.5,color=NAVY); ax.set_ylim(-0.08,1.1)
ax=axs[1]; sg=np.linspace(0,12,300); mm=2*ysep-1; Ju=np.array([np.logaddexp(0,-mm*s_*(hours-3.5)).sum() for s_ in sg]); Jp=Ju+0.5*sg**2
ax.plot(sg,Ju,color=RED,lw=1.8,label='$J$, no penalty ($C=\\infty$)'); ax.plot(sg,Jp,color=GREEN,lw=1.8,ls='--',label='$J$ with $\\frac{1}{2}w^2$, $C$ = 1'); kmin=int(np.argmin(Jp)); ax.plot(sg[kmin],Jp[kmin],'*',color=OR,ms=11,mec='white',zorder=6)
ax.text(4.3,3.4,f'minimum of the\ndashed curve:\n$w$ = {sg[kmin]:.1f}',fontsize=8.5,color=NAVY,va='center')
ax.set_xlabel('weight  $w$  (boundary fixed at 3.5)'); ax.set_ylabel('$J$'); ax.set_ylim(0,8); ax.set_xlim(0,14); ax.legend(loc='upper right',fontsize=8.2); ax.grid(color=GRID,lw=0.5); ax.set_title('no minimum without a penalty',fontsize=9.5,color=NAVY)
NUM['sep']={'wmin_C1_line':round(float(sg[kmin]),2)}
fig.tight_layout(w_pad=1.0); sv(fig,'f1611.png',450)

# ---- 16.12-1 case study figure
pb=make_pipeline(StandardScaler(),LogisticRegression()).fit(Xb_tr,yb_tr); pr=pb.predict(Xb_te); cm=confusion_matrix(yb_te,pr); wb=pb[-1].coef_[0]
fig,axs=plt.subplots(1,2,figsize=(4.9,2.5),dpi=240,gridspec_kw={'width_ratios':[1,1.45]}); ax=axs[0]
ax.imshow(cm,cmap=LinearSegmentedColormap.from_list('w',['#FFFFFF','#9DBCE8']),vmin=0,vmax=cm.max()*1.3)
for i_ in range(2):
    for j_ in range(2): ax.text(j_,i_,str(cm[i_,j_]),ha='center',va='center',fontsize=13,color=NAVY if i_==j_ else RED,weight='bold')
ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['malignant','benign'],fontsize=8.5); ax.set_yticklabels(['malignant','benign'],fontsize=8.5,rotation=90,va='center')
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.set_title('test set, 143 patients',fontsize=9.5,color=NAVY)
for sp_ in ax.spines.values(): sp_.set_visible(False)
ax.tick_params(length=0)
ax=axs[1]; sel=np.r_[np.argsort(wb)[:4],np.argsort(wb)[-4:]]; sel=sel[np.argsort(wb[sel])]
ax.barh([bc.feature_names[j] for j in sel],wb[sel],color=[BLUE if wb[j]<0 else OR for j in sel],height=0.62); ax.axvline(0,color=GREY,lw=0.7)
ax.set_xlabel('weight per standard deviation'); ax.set_title('largest weights',fontsize=9.5,color=NAVY); ax.tick_params(axis='y',labelsize=8); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=0.8); sv(fig,'f1612.png',470)
NUM['case']={'cm':cm.tolist()}
json.dump({'NUM':NUM},open('verify/fig_numbers.json','w'),indent=1)
json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
bad={k:v for k,v in CHK.items() if v['min_font_printed']<7.5}; print('figures below 7.5 pt:',bad); print(json.dumps(NUM))

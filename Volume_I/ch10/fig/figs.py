# All figures of Chapter 10. Run: python3 fig/figs.py  (numbers -> verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
NUM={}
x1=np.array([1.,2,3,4,5]); x2=np.array([1.,2,3,5,4]); y=np.array([3.,4,6,7,10]); X=np.c_[x1,x2]
Xc=X-X.mean(0); yc=y-y.mean(); G=Xc.T@Xc; g=Xc.T@yc
def wr(a,yv=None):
    gg=g if yv is None else Xc.T@(yv-yv.mean()); return np.linalg.solve(G+a*np.eye(2),gg)
w0=wr(0); sse0=float(((yc-Xc@w0)**2).sum())
def Jols(W1,W2):
    d1=W1-w0[0]; d2=W2-w0[1]; return sse0+G[0,0]*d1*d1+2*G[0,1]*d1*d2+G[1,1]*d2*d2
def Jpen(W1,W2,a): return Jols(W1,W2)+a*(W1**2+W2**2)
def axes(ax,xl,yl): ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
g1,g2=np.meshgrid(np.linspace(-1.2,3.6,300),np.linspace(-1.6,1.6,300))

# ---- 10.0-1 opener
fig,ax=plt.subplots(figsize=(5.4,2.7),dpi=240)
lev=sse0+np.array([0.3,1,2.5,5,9])
ax.contour(g1,g2,Jols(g1,g2),levels=lev,colors=BLUE,linewidths=0.8,alpha=0.8)
al=np.logspace(-2,3,200); P=np.array([wr(a) for a in al]); ax.plot(P[:,0],P[:,1],color=GREEN,lw=2,zorder=4)
wr1=wr(1.0); rad=np.hypot(*wr1); th=np.linspace(0,2*np.pi,200); ax.plot(rad*np.cos(th),rad*np.sin(th),color=PUR,lw=1.2,ls='--',zorder=3)
ax.contour(g1,g2,Jols(g1,g2),levels=[Jols(*wr1)],colors=BLUE,linewidths=1.6,zorder=3)
ax.plot(*w0,'o',color=BLUE,ms=7,mec='white',zorder=6); ax.text(2.2,-1.45,'least squares ($\\alpha$ = 0)',color=BLUE,fontsize=9.5,ha='center',bbox=dict(fc='white',ec='none',pad=1.5,alpha=0.9))
ax.plot(*wr1,'*',color=GREEN,ms=13,mec='white',zorder=6); ax.text(wr1[0]+0.2,wr1[1]+0.3,'ridge, $\\alpha$ = 1',color=GREEN,fontsize=9.5,weight='bold',bbox=dict(fc='white',ec='none',pad=1.5,alpha=0.9))
for a,dx,dy in [(10,-0.05,0.17),(100,0.08,-0.3)]:
    p=wr(a); ax.plot(*p,'o',color=GREEN,ms=4.5,zorder=6); ax.text(p[0]+dx,p[1]+dy,f'$\\alpha$ = {a}',color=GREEN,fontsize=9,bbox=dict(fc='white',ec='none',pad=1,alpha=0.9))
ax.plot(0,0,'o',color=NAVY,ms=4,zorder=6); ax.text(-0.1,0.12,'$\\alpha\\to\\infty$',color=NAVY,fontsize=9,ha='right')
ax.text(-1.1,-1.0,'circle: $\\Vert\\mathbf{w}\\Vert$ = 1.53\n(dashed)',color=PUR,fontsize=9,va='top',bbox=dict(fc='white',ec='none',pad=1.5,alpha=0.9))
ax.text(-1.1,1.45,'ellipses: equal SSE',color=BLUE,fontsize=9,va='top')
ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
ax.set_xlim(-1.2,3.6); ax.set_ylim(-1.6,1.6); ax.set_aspect('equal'); axes(ax,'weight $w_1$ (odometer)','weight $w_2$ (planner)'); save(fig,'f1000.png')

# ---- 10.2-1 data and two bowls
fig,axs=plt.subplots(1,3,figsize=(6.4,2.4),dpi=240)
ax=axs[0]; ax.plot([0,6],[0,6],color=GRID,lw=1,ls='--'); ax.scatter(x1,x2,s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
for a,b_,t in zip(x1,x2,y): ax.text(a+0.18,b_-0.1,f'{t:.0f}',fontsize=9,color=GREY,va='top')
ax.set_xlim(0,6); ax.set_ylim(0,6); ax.set_aspect('equal'); axes(ax,'odometer $x_1$ (km)','planner $x_2$ (km)'); ax.set_title('Data (numbers: time $y$)',fontsize=9.5,color=NAVY)
for ax,a,t in [(axs[1],0,'SSE only'),(axs[2],1,'SSE + 1 $\\cdot\\Vert\\mathbf{w}\\Vert^2$')]:
    Z=Jpen(g1,g2,a); lv=np.array([0.3,1,2,4,8,16,32,64,128,256,512])+Z.min()
    ax.contourf(g1,g2,Z,levels=np.concatenate([[Z.min()],lv,[1e3]]),cmap='Blues_r',alpha=0.95)
    ax.contour(g1,g2,Z,levels=lv,colors='white',linewidths=0.5)
    m=wr(a); ax.plot(*m,'*',color=GREEN if a else OR,ms=13,mec='white',zorder=5); ax.plot(0,0,'+',color=NAVY,ms=7)
    ax.set_xlim(-1.2,3.4); ax.set_ylim(-1.6,1.6); ax.set_aspect('equal'); axes(ax,'$w_1$','$w_2$'); ax.set_title(t,fontsize=9.5,color=NAVY)
    ax.text(m[0]-0.1,m[1]-0.62,f'({m[0]:.3f}, {m[1]:.3f})',fontsize=9,color='white',ha='center',bbox=dict(fc=NAVY,ec='none',pad=1.2,alpha=0.85))
fig.tight_layout(w_pad=0.6); save(fig,'f1021.png')

# ---- 10.4-1 filter factors and df
dd=load_diabetes(); Xd,yd=dd.data,dd.target; sd=np.linalg.svd(Xd-Xd.mean(0),compute_uv=False)
fig,axs=plt.subplots(1,2,figsize=(6.0,2.5),dpi=240); al=np.logspace(-3,3,300)
ax=axs[0]
for lam,c,lab in [(19,BLUE,'strong direction, $\\sigma^2$ = 19'),(1,OR,'weak direction, $\\sigma^2$ = 1')]:
    ax.plot(al,lam/(lam+al),color=c,lw=1.8,ls='-' if lam==19 else '--',label=lab)
ax.axvline(1,color=PUR,lw=1,ls=':'); ax.plot([1,1],[0.95,0.5],'o',color=PUR,ms=4); ax.text(40,0.62,'$\\alpha$ = 1:\n0.95 and 0.50',fontsize=9,color=PUR)
ax.set_xscale('log'); axes(ax,'penalty $\\alpha$','filter factor $f_j$'); ax.legend(fontsize=8.5,loc='lower left',frameon=True,facecolor='white',edgecolor='none',framealpha=0.9); ax.set_ylim(0,1.05); ax.set_title('Shrinkage of each direction',fontsize=9.5,color=NAVY)
ax=axs[1]; s2=np.array([19.,1.]); ax.plot(al,[(s2/(s2+a)).sum() for a in al],color=BLUE,lw=1.8,label='toy ($p$ = 2)')
ax.plot(al,[(sd**2/(sd**2+a)).sum() for a in al],color=GREEN,lw=1.8,ls='--',label='diabetes ($p$ = 10)'); ax.set_xscale('log'); axes(ax,'penalty $\\alpha$','degrees of freedom df($\\alpha$)')
ax.legend(fontsize=8.5,loc='upper right'); ax.set_title('Effective number of weights',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1041.png')
NUM['diab_sv2']=[round(float(v),4) for v in sd**2]; NUM['diab_cond']=round(float(sd[0]**2/sd[-1]**2),1)
NUM['diab_df']={str(a):round(float((sd**2/(sd**2+a)).sum()),2) for a in [0.001,0.01,0.05,0.3,1,10]}
NUM['toy_cond']={'ols':19.0,'ridge1':10.0}

# ---- 10.6-1 flow chart
fig,ax=new(6.0,4.9)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=9): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
fb(3.5,6.0,5.2,0.5,'Input:  $\\mathbf{X}$,  $\\mathbf{y}$,  $\\alpha$  (and sample weights)',fc=LGREY,ec=GREY)
fb(3.5,5.2,5.2,0.5,'Validate; rescale rows if sample_weight is given')
fb(3.5,4.3,5.2,0.75,'If fit_intercept: centre the data\n$\\mathbf{X}_c=\\mathbf{X}-\\bar{\\mathbf{x}}$,   $\\mathbf{y}_c=\\mathbf{y}-\\bar y$   (Eq. 10.3-3)')
ax.add_patch(Polygon([(3.5,3.75),(4.8,3.35),(3.5,2.95),(2.2,3.35)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.5,3.35,'solver?',ha='center',va='center',fontsize=9.5,color=NAVY)
fb(0.95,1.95,1.75,1.35,'cholesky\n(auto, dense)\n$p\\leq n$: $\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}+\\alpha\\mathbf{I}$\n$p>n$: Eq. 10.3-8',fc='#E3F4EA',ec=GREEN,fs=8.6)
fb(2.85,1.95,1.6,1.35,'svd\nEq. 10.4-3\nthe most stable',fc=LB,ec=BLUE,fs=8.6)
fb(4.75,1.95,1.8,1.35,'lsqr, sparse_cg,\nsag, saga\niterate until tol',fc='#F1EAF7',ec=PUR,fs=8.6)
fb(6.55,1.95,1.5,1.35,'lbfgs\nonly with\npositive=True',fc='#F1EAF7',ec=PUR,fs=8.6)
fb(3.75,0.55,5.6,0.5,'Intercept:  $b=\\bar y-\\bar{\\mathbf{x}}^{\\mathrm{T}}\\hat{\\mathbf{w}}$   (Eq. 10.3-2)')
fb(3.75,-0.25,5.6,0.5,'Store  coef_, intercept_, n_iter_, solver_',fc=LGREY,ec=GREY)
A((3.5,5.75),(3.5,5.45)); A((3.5,4.95),(3.5,4.68)); A((3.5,3.92),(3.5,3.76))
ax.plot([3.5,3.5],[2.95,2.8],color=NAVY,lw=0.9); ax.plot([0.95,6.55],[2.8,2.8],color=NAVY,lw=0.9)
for xx in [0.95,2.85,4.75,6.55]: A((xx,2.8),(xx,2.63)); ax.plot([xx,xx],[1.27,0.95],color=NAVY,lw=0.9)
ax.plot([0.95,6.55],[0.95,0.95],color=NAVY,lw=0.9); A((3.75,0.95),(3.75,0.8)); A((3.75,0.3),(3.75,0.0))
ax.set_xlim(-0.3,7.5); ax.set_ylim(-0.6,6.4); save(fig,'f1061.png')

# ---- 10.10-1 collinearity gallery
rng=np.random.default_rng(1)
def batch(rho):
    D=[]
    for k in range(300):
        z=rng.normal(size=(50,2)); a_=z[:,0]; b_=rho*z[:,0]+np.sqrt(1-rho**2)*z[:,1]; D.append((np.c_[a_,b_],a_+b_+rng.normal(0,1,50)))
    return D
D0=batch(0.0); D1=batch(0.999); NUM['collin']={}
def est(D,a): return np.array([(LinearRegression() if a==0 else Ridge(alpha=a)).fit(Xx,yy).coef_ for Xx,yy in D])
NUM['collin']['indep_ols_sd']=round(float(est(D0,0)[:,0].std()),2)
for a in [0,0.1,1,10,100]:
    W=est(D1,a); NUM['collin'][str(a)]={'mean_w1':round(float(W[:,0].mean()),3),'mean_w2':round(float(W[:,1].mean()),3),'sd_w1':round(float(W[:,0].std()),3),
      'mean_sum':round(float(W.sum(1).mean()),3),'sd_sum':round(float(W.sum(1).std()),3),'rms':round(float(np.sqrt(((W-1)**2).sum(1).mean())),3)}
fig,axs=plt.subplots(2,2,figsize=(5.6,4.7),dpi=240)
for ax,a in zip(axs.ravel(),[0,1,10,100]):
    W=est(D1,a); ax.scatter(W[:,0],W[:,1],s=7 if a==0 else 10,color=PUR,alpha=0.55,lw=0); ax.plot(1,1,'*',color=GREEN,ms=11,mec='white',zorder=6)
    ax.axvline(1,color=GREEN,lw=0.6); ax.axhline(1,color=GREEN,lw=0.6); ax.grid(color=GRID,lw=0.5)
    ax.set_title(f'$\\alpha$ = {a}' if a else '$\\alpha$ = 0 (least squares)',fontsize=10,color=NAVY)
    ax.text(0.04,0.05,f'sd of $w_1$ = {W[:,0].std():.2f}',transform=ax.transAxes,fontsize=9.5,color=PUR,bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
axs[0,0].set_xlim(-11,13); axs[0,0].set_ylim(-11,13)
for ax in axs.ravel()[1:]: ax.set_xlim(0,2.2); ax.set_ylim(0,2.2); ax.set_aspect('equal')
for ax in axs.ravel(): ax.set_xlabel('fitted $w_1$'); ax.set_ylabel('fitted $w_2$')
fig.tight_layout(h_pad=0.8,w_pad=0.6); save(fig,'f1101.png')

# ---- 10.10-2 path and validation curve
Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.25,random_state=0)
al=np.logspace(-3,1,41); path=np.array([Ridge(alpha=a).fit(Xtr,ytr).coef_ for a in al])
cvs=np.array([cross_val_score(Ridge(alpha=a),Xtr,ytr,cv=5,scoring='r2') for a in al])
trs=np.array([Ridge(alpha=a).fit(Xtr,ytr).score(Xtr,ytr) for a in al]); tes=np.array([Ridge(alpha=a).fit(Xtr,ytr).score(Xte,yte) for a in al])
acv=RidgeCV(alphas=al).fit(Xtr,ytr).alpha_
NUM['val']={'alpha_loo':round(float(acv),4),'best_cv_alpha':round(float(al[cvs.mean(1).argmax()]),4),'best_cv_r2':round(float(cvs.mean(1).max()),3),'cv_r2_at_0.001':round(float(cvs.mean(1)[0]),3),
  'test_best_alpha':round(float(al[tes.argmax()]),4),'test_best':round(float(tes.max()),3),'test_at_loo':round(float(Ridge(alpha=acv).fit(Xtr,ytr).score(Xte,yte)),3),'test_at_10':round(float(tes[-1]),3),'train_at_10':round(float(trs[-1]),3)}
fig,axs=plt.subplots(1,2,figsize=(6.0,2.8),dpi=240)
ax=axs[0]; cols=plt.cm.tab10(np.arange(10))
for j,nm in enumerate(dd.feature_names): ax.plot(al,path[:,j],color=cols[j],lw=1.4,ls=['-','--','-.',':'][j%4])
ax.set_xscale('log'); ax.axvline(acv,color=PUR,lw=1.1,ls='--'); ax.axhline(0,color=GREY,lw=0.6); axes(ax,'penalty $\\alpha$','coefficient $w_j$'); ax.set_title('Coefficient paths (training data)',fontsize=9.5,color=NAVY)
offs={'s5':45,'bmi':-45,'s2':40,'s1':30,'s3':-45}
for nm,o in offs.items():
    j=list(dd.feature_names).index(nm); ax.text(0.00112,path[0,j]+o,nm,fontsize=9,color=cols[j],weight='bold',va='center',bbox=dict(fc='white',ec='none',pad=0.6,alpha=0.8))
ax.set_xlim(0.001,10)
ax=axs[1]; ax.plot(al,trs,color=BLUE,lw=1.6,label='training'); ax.plot(al,cvs.mean(1),color=GREEN,lw=1.8,ls='--',label='5-fold (training part)'); ax.plot(al,tes,color=GREY,lw=1.4,ls=':',label='test')
ax.axvline(acv,color=PUR,lw=1.1,ls='--'); ax.text(acv*1.2,0.585,f'LOO choice $\\alpha$ = {acv:.3f}',fontsize=9,color=PUR,va='bottom')
ax.set_xscale('log'); axes(ax,'penalty $\\alpha$','$R^2$'); ax.set_ylim(0.1,0.66); ax.legend(fontsize=8.5,loc='lower left'); ax.set_title('Validation curve',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1102.png')

# ---- 10.12-1 case study
ols=LinearRegression().fit(Xtr,ytr); rcv=RidgeCV(alphas=np.logspace(-3,1,41)).fit(Xtr,ytr); po,pr=ols.predict(Xte),rcv.predict(Xte)
NUM['case']={'max_pred_diff':round(float(np.abs(po-pr).max()),2),'corr_pred':round(float(np.corrcoef(po,pr)[0,1]),4),'cond_train_corr':None}
Ct=np.linalg.eigvalsh(np.corrcoef(Xd.T)); NUM['case']['corr_eig_min']=round(float(Ct[0]),4); NUM['case']['corr_eig_max']=round(float(Ct[-1]),3)
Xt=Xtr-Xtr.mean(0); sv=np.linalg.svd(Xt,compute_uv=False)**2; NUM['case']['cond_ols']=round(float(sv[0]/sv[-1]),1); NUM['case']['cond_ridge']=round(float((sv[0]+rcv.alpha_)/(sv[-1]+rcv.alpha_)),1)
NUM['case']['col_sumsq']=round(float((Xd**2).sum(0)[0]),3)
fig,axs=plt.subplots(1,2,figsize=(6.0,2.9),dpi=240,gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]; idx=np.arange(10); ax.barh(idx+0.2,ols.coef_,height=0.38,color=BLUE,label='least squares'); ax.barh(idx-0.2,rcv.coef_,height=0.38,color=GREEN,hatch='///',edgecolor='white',label='ridge')
ax.set_yticks(idx); ax.set_yticklabels(dd.feature_names); ax.invert_yaxis(); ax.axvline(0,color=GREY,lw=0.7); ax.set_xlabel('coefficient $w_j$'); ax.legend(fontsize=8.5,loc='lower left'); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.set_title('Fitted coefficients',fontsize=9.5,color=NAVY)
ax=axs[1]; ax.plot([50,300],[50,300],color=GREEN,lw=1.2); ax.scatter(po,pr,s=13,color=GREY,alpha=0.8,lw=0); ax.set_aspect('equal'); axes(ax,'least squares prediction','ridge prediction'); ax.set_title('Test predictions',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1121.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

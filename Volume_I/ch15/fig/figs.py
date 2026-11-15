# All main figures of Chapter 15. Run: python3 fig/figs.py   (writes PNGs into ch15/ and numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from math import comb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, validation_curve, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.datasets import load_diabetes
NUM={}
FS=9
def grid_(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx,yy,s=24,c=BLUE,**k): ax.scatter(xx,yy,s=s,color=c,edgecolor='white',lw=0.6,zorder=5,**k)
RMSE=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))

# ---------------- toy data (battery use against speed)
xt=np.array([1.,2,3,4,5]); yt=np.array([2.,3,8,15,22])
# ---------------- generated curved data (same as code/val_curve.py)
def curve(x): return 4*np.exp(-(x-3)**2/0.8)+0.5*x
rng=np.random.default_rng(0)
x=np.sort(rng.uniform(0,10,40))[:,None]; y=curve(x[:,0])+rng.normal(0,0.4,40)
x_new=rng.uniform(0,10,1000)[:,None]; y_new=curve(x_new[:,0])+rng.normal(0,0.4,1000)
gx=np.linspace(0,10,400)[:,None]
def polym(d,alpha=None): return make_pipeline(StandardScaler(),PolynomialFeatures(d,include_bias=False),LinearRegression())
def splm(k,deg=3): return make_pipeline(SplineTransformer(n_knots=k,degree=deg,include_bias=False),LinearRegression())

# ================= f1500 opener: a curve as a weighted sum of bumps
fig,axs=plt.subplots(2,1,figsize=(5.0,2.9),dpi=240,sharex=True,gridspec_kw={'height_ratios':[1.6,1],'hspace':0.12})
st=SplineTransformer(n_knots=8,degree=3).fit(x); Phi=st.transform(x); m=LinearRegression(fit_intercept=False).fit(Phi,y)
G=st.transform(gx); w=m.coef_; ax=axs[0]
ax.plot(gx[:,0],curve(gx[:,0]),color=GREY,lw=0.9,ls=':',zorder=2); ax.plot(gx[:,0],G@w,color=GREEN,lw=2.2,zorder=4); pts(ax,x[:,0],y,s=16)
ax.set_ylabel('$y$',fontsize=FS+1); ax.set_xticks([]); ax.set_xlim(0,10); ax.set_ylim(-0.8,7.6)
ax.text(5.9,6.7,'sum of the pieces below',color=GREEN,fontsize=FS+0.5,weight='bold'); ax.text(0.3,6.7,'data',color=BLUE,fontsize=FS+0.5); ax.text(6.0,1.5,'true curve (dotted)',color=GREY,fontsize=FS)
ax=axs[1]; cm=[PUR,TEAL,MAG,OR,BLUE,RED,GREEN]
for j in range(G.shape[1]):
    ax.fill_between(gx[:,0],0,w[j]*G[:,j],color=PUR,alpha=0.10+0.0*j,lw=0); ax.plot(gx[:,0],w[j]*G[:,j],color=PUR,lw=1.1,ls='-' if j%2==0 else '--')
kn=st.bsplines_[0].t[3:-3]; ax.plot(kn,np.zeros_like(kn),'|',color=NAVY,ms=7,mew=1.2,clip_on=False)
ax.set_ylabel('$w_j\\,B_j(x)$',fontsize=FS+1); ax.set_xlabel('feature $x$   (ticks mark the knots)',fontsize=FS+1); ax.set_xlim(0,10); ax.set_ylim(0,8.6)
ax.text(5.6,7.3,'each bump: a fixed basis function\ntimes one weight $w_j$',fontsize=FS,color=PUR,va='center')
save(fig,'f1500.png')
NUM['opener_w']=[round(float(v),2) for v in w]

# ================= f1511 the toy data and a line
ml=LinearRegression().fit(xt[:,None],yt); pl=ml.predict(xt[:,None]); rl=yt-pl
NUM['toy_line']={'w':round(float(ml.coef_[0]),2),'b':round(float(ml.intercept_),2),'R2':round(float(ml.score(xt[:,None],yt)),4),'SSE':round(float(rl@rl),2)}
fig,axs=plt.subplots(1,2,figsize=(4.4,2.15),dpi=240); ax=axs[0]
xx=np.array([0.3,5.7]); ax.plot(xx,ml.intercept_+ml.coef_[0]*xx,color=GREEN,lw=1.8)
for a,b_,c in zip(xt,yt,pl): ax.plot([a,a],[b_,c],color=RED,lw=1.1)
pts(ax,xt,yt,s=30); grid_(ax); ax.set_xlabel('speed $x$ (5 km/h)',fontsize=FS); ax.set_ylabel('battery used $y$ (%)',fontsize=FS); ax.set_title(f'$R^2$ = {NUM["toy_line"]["R2"]:.3f}',fontsize=FS+0.5,color=NAVY)
ax.set_xlim(0.3,5.7)
ax=axs[1]; ax.axhline(0,color=GREEN,lw=1); ax.vlines(xt,0,rl,color=RED,lw=1.4); ax.scatter(xt,rl,s=26,color=RED,zorder=5,marker='s')
for a,v in zip(xt,rl): ax.text(a,v+(0.35 if v>=0 else -0.35),f'{v:+.1f}'.replace('-','\u2212'),ha='center',va='bottom' if v>=0 else 'top',fontsize=FS-0.5,color=RED)
grid_(ax); ax.set_ylim(-3.6,3.6); ax.set_xlim(0.3,5.7); ax.set_xlabel('speed $x$ (5 km/h)',fontsize=FS); ax.set_ylabel('residual $r$ (%)',fontsize=FS); ax.set_title('signs:  +  −  −  −  +',fontsize=FS+0.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1511.png')

# ================= f1521 a model is a weighted sum of columns
w3=np.linalg.solve(np.c_[np.ones(5),xt,xt**2].T@np.c_[np.ones(5),xt,xt**2],np.c_[np.ones(5),xt,xt**2].T@yt); NUM['toy_w']=[round(float(v),4) for v in w3]
xx=np.linspace(0,6,200)
fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240,sharey=False)
ax=axs[0]
for f,lab,c,ls in ((np.ones_like(xx),'$1$',GREY,'-'),(xx,'$x$',BLUE,'--'),(xx**2,'$x^2$',PUR,'-.')): ax.plot(xx,f,color=c,lw=1.6,ls=ls,label=lab)
ax.legend(fontsize=FS,loc='upper left',handlelength=2.2); grid_(ax); ax.set_xlabel('$x$',fontsize=FS); ax.set_ylim(-2,38); ax.set_title('the three columns',fontsize=FS+0.5,color=NAVY)
ax=axs[1]
for f,lab,c,ls in ((w3[0]*np.ones_like(xx),'$1.4\\times 1$',GREY,'-'),(w3[1]*xx,'$-0.8\\times x$',BLUE,'--'),(w3[2]*xx**2,'$1.0\\times x^2$',PUR,'-.')): ax.plot(xx,f,color=c,lw=1.3,ls=ls,label=lab)
ax.plot(xx,w3[0]+w3[1]*xx+w3[2]*xx**2,color=GREEN,lw=2.2,label='sum'); pts(ax,xt,yt,s=28); ax.legend(fontsize=FS-0.5,loc='upper left',handlelength=2.2); grid_(ax); ax.set_xlabel('$x$',fontsize=FS); ax.set_ylim(-5,38); ax.set_title('weighted, then added',fontsize=FS+0.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1521.png')

# ================= f1531 conditioning of monomials
fig,axs=plt.subplots(1,2,figsize=(4.7,2.2),dpi=240); ax=axs[0]; t=np.linspace(0,1,200)
for k,c in zip(range(1,7),[BLUE,TEAL,GREEN,OR,RED,PUR]): ax.plot(t,t**k,color=c,lw=1.3,ls=['-','--'][k%2])
ax.text(0.04,0.93,'$x, x^2, \\dots, x^6$',fontsize=FS,color=NAVY,va='top'); grid_(ax); ax.set_xlabel('$x$',fontsize=FS); ax.set_title('monomials look alike',fontsize=FS+0.5,color=NAVY)
ax=axs[1]; ds=list(range(3,16)); xr=np.linspace(0,10,60); xs=np.linspace(-1,1,60); cr=[];cs=[];cb=[]
for d in ds:
    cr.append(np.linalg.cond(PolynomialFeatures(d).fit_transform(xr[:,None]))); cs.append(np.linalg.cond(PolynomialFeatures(d).fit_transform(xs[:,None])))
    cb.append(np.linalg.cond(SplineTransformer(n_knots=d-1,degree=3).fit_transform(xr[:,None])))
ax.semilogy(ds,cr,'-o',color=RED,ms=3,lw=1.3,label='$x\\in[0,10]$'); ax.semilogy(ds,cs,'--s',color=BLUE,ms=3,lw=1.3,label='$x\\in[-1,1]$'); ax.semilogy(ds,cb,':^',color=GREEN,ms=3,lw=1.5,label='B-splines')
ax.axhline(1/np.finfo(float).eps,color=GREY,lw=0.8,ls='-.'); ax.text(3.1,1/np.finfo(float).eps/2.5,'limit of double precision',fontsize=FS-0.5,color=GREY,ha='left',va='top'); ax.set_ylim(1,1e22); ax.set_yticks([1e0,1e5,1e10,1e15,1e20])
ax.text(3.2,1e10,'$x\\in[0,10]$',fontsize=FS-0.5,color=RED,ha='left'); ax.text(7.2,1e7,'$x\\in[-1,1]$',fontsize=FS-0.5,color=BLUE,ha='left'); ax.text(10.5,1e2,'B-splines',fontsize=FS-0.5,color=GREEN,ha='left')
grid_(ax); ax.set_xlabel('degree $d$',fontsize=FS); ax.set_title('condition number',fontsize=FS+0.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1531.png')
NUM['cond']={str(d):{'raw':float(a),'scaled':float(b),'spline':float(c)} for d,a,b,c in zip(ds,cr,cs,cb)}

# ================= f1532 Runge
fr=lambda z:1/(1+25*z**2); xr_=np.linspace(-1,1,11); zz=np.linspace(-1,1,400)
pm=np.polyfit(xr_,fr(xr_),10); pz=np.polyval(pm,zz)
sm_=make_pipeline(SplineTransformer(n_knots=8,degree=3,include_bias=False),LinearRegression()).fit(xr_[:,None],fr(xr_)); sz=sm_.predict(zz[:,None])
NUM['runge']={'poly_maxerr':round(float(np.abs(pz-fr(zz)).max()),3),'poly_maxerr_at':round(float(zz[np.abs(pz-fr(zz)).argmax()]),2),'spline_maxerr':round(float(np.abs(sz-fr(zz)).max()),3),'poly_max_value':round(float(pz.max()),3)}
fig,axs=plt.subplots(1,2,figsize=(4.7,2.1),dpi=240,sharey=True)
for ax,pz_,c,t_,ls in ((axs[0],pz,RED,'degree 10 through 11 points','-'),(axs[1],sz,GREEN,'cubic spline, 8 knots','-')):
    ax.plot(zz,fr(zz),color=GREY,lw=1.0,ls=':'); ax.plot(zz,pz_,color=c,lw=1.6); pts(ax,xr_,fr(xr_),s=22); grid_(ax); ax.set_xlabel('$x$',fontsize=FS); ax.set_title(t_,fontsize=FS,color=NAVY); ax.set_ylim(-0.6,1.25)
axs[0].set_ylabel('$f(x)$',fontsize=FS)
axs[0].text(-0.97,-0.45,f'error {NUM["runge"]["poly_maxerr"]:.2f}',fontsize=FS-0.5,color=RED); axs[1].text(-0.97,-0.45,f'error {NUM["runge"]["spline_maxerr"]:.2f}',fontsize=FS-0.5,color=GREEN)
fig.tight_layout(w_pad=0.6); save(fig,'f1532.png')

# ================= f1541 B-spline bases of degree 0..3
fig,axs=plt.subplots(2,2,figsize=(4.9,3.3),dpi=240,sharex=True,sharey=True); zz=np.linspace(0,5,500)[:,None]; cols=[BLUE,OR,GREEN,PUR,RED,TEAL,MAG,NAVY,GREY]
for ax,d in zip(axs.ravel(),range(4)):
    s=SplineTransformer(n_knots=6,degree=d).fit(np.linspace(0,5,6)[:,None]); B=s.transform(zz)
    B=np.where(B>1e-12,B,np.nan)
    for j in range(B.shape[1]):
        ax.plot(zz[:,0],B[:,j],color=cols[j%len(cols)],lw=1.4,ls='-' if j%2==0 else '--'); ax.fill_between(zz[:,0],0,np.nan_to_num(B[:,j]),color=cols[j%len(cols)],alpha=0.15,lw=0)
    ax.plot(zz[:,0],np.nansum(B,1),color='black',lw=0.7,ls=':'); ax.plot(np.arange(6),np.zeros(6),'|',color=NAVY,ms=6,mew=1.2,clip_on=False)
    ax.set_title(f'degree {d}:  {B.shape[1]} functions',fontsize=FS,color=NAVY); ax.set_ylim(-0.04,1.14); grid_(ax)
for ax in axs[1]: ax.set_xlabel('$x$',fontsize=FS)
for ax in axs[:,0]: ax.set_ylabel('$B_j(x)$',fontsize=FS)
axs[0,1].text(2.5,1.03,'sum = 1',fontsize=FS-0.5,color='black',ha='center',va='bottom')
fig.tight_layout(w_pad=0.6,h_pad=0.8); save(fig,'f1541.png')

# ================= f1542 extrapolation and periodic
xs_=np.linspace(0,10,25)[:,None]; ys_=np.sin(xs_[:,0]/2)+0.1*np.cos(7*xs_[:,0]); ge=np.linspace(-3,13,300)[:,None]
fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240); ax=axs[0]
for mode,c,ls in (('constant',GREEN,'-'),('linear',OR,'--'),('continue',RED,':')):
    sp=SplineTransformer(n_knots=5,extrapolation=mode).fit(xs_); f=LinearRegression().fit(sp.transform(xs_),ys_)
    ax.plot(ge[:,0],f.predict(sp.transform(ge)),color=c,lw=1.6,ls=ls,label=mode)
pts(ax,xs_[:,0],ys_,s=12); ax.axvspan(-3,0,color=LGREY,zorder=0); ax.axvspan(10,13,color=LGREY,zorder=0); ax.set_ylim(-3,2.2); ax.set_xlim(-3,13); ax.legend(fontsize=FS-0.5,loc='lower center',handlelength=2.2,ncol=1,borderaxespad=0.2); grid_(ax)
ax.set_xlabel('$x$ (grey: outside the data)',fontsize=FS-0.5); ax.set_title('extrapolation',fontsize=FS+0.5,color=NAVY)
def circ(h,c): d=np.abs(h-c); return np.minimum(d,24-d)
def demand(h): return 2+5*np.exp(-circ(h,8)**2/4)+3*np.exp(-circ(h,17.5)**2/6)
r2=np.random.default_rng(0); hh=r2.uniform(0,24,120)[:,None]; yh=demand(hh[:,0])+r2.normal(0,0.4,120); knots=np.linspace(0,24,9)[:,None]; gh=np.linspace(-5,29,400)[:,None]
ax=axs[1]
for nm,sp,c,ls in (('ordinary',SplineTransformer(knots=knots),RED,'--'),('periodic',SplineTransformer(knots=knots,extrapolation='periodic'),GREEN,'-')):
    f=make_pipeline(sp,Ridge(alpha=0.01)).fit(hh,yh); ax.plot(gh[:,0],f.predict(gh),color=c,lw=1.6,ls=ls,label=nm)
pts(ax,hh[:,0],yh,s=8); ax.axvspan(-5,0,color=LGREY,zorder=0); ax.axvspan(24,29,color=LGREY,zorder=0); ax.set_xlim(-5,29); ax.set_ylim(-1.6,8.2); ax.set_xticks([0,12,24]); grid_(ax); ax.legend(fontsize=FS-0.5,loc='lower center',handlelength=2.2,ncol=2,columnspacing=1.0,borderaxespad=0.1); ax.set_xlabel('hour of day (grey: wrapped around)',fontsize=FS-0.5); ax.set_title('periodic splines',fontsize=FS+0.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f1542.png')

# ================= f1591 gallery
fig,axs=plt.subplots(2,3,figsize=(5.2,3.45),dpi=240,sharex=True,sharey=True); cfg=[('degree',d) for d in (1,10,20)]+[('knots',k) for k in (3,11,20)]
NUM['gallery']={}
for ax,(kind,v) in zip(axs.ravel(),cfg):
    m=polym(v) if kind=='degree' else splm(v); m.fit(x,y); rt=RMSE(m.predict(x),y); re_=RMSE(m.predict(x_new),y_new)
    NUM['gallery'][f'{kind}{v}']={'train':round(rt,3),'test':round(re_,3)}
    ax.plot(gx[:,0],curve(gx[:,0]),color=GREY,lw=0.9,ls=':',zorder=2); ax.plot(gx[:,0],np.clip(m.predict(gx),-3,12),color=GREEN,lw=1.6,zorder=4); pts(ax,x[:,0],y,s=9)
    ax.set_title(('polynomial, degree ' if kind=='degree' else 'spline, n_knots = ')+str(v),fontsize=FS-0.5,color=NAVY); ax.set_ylim(-1.5,8.5); ax.set_xlim(0,10); grid_(ax)
    ax.text(9.8,-0.9,f'test RMSE {re_:.2f}' if re_<10 else f'test RMSE {re_:.1f}',fontsize=FS-0.5,color=RED,ha='right')
for ax in axs[1]: ax.set_xlabel('$x$',fontsize=FS)
for ax in axs[:,0]: ax.set_ylabel('$y$',fontsize=FS)
fig.tight_layout(w_pad=0.5,h_pad=0.8); save(fig,'f1591.png')

# ================= f1592 validation curves
cv=KFold(5,shuffle=True,random_state=0); NUM['val']={}
fig,axs=plt.subplots(1,3,figsize=(5.0,2.2),dpi=240)
spl20=make_pipeline(SplineTransformer(n_knots=20,include_bias=False),Ridge())
for ax,(model,name,vals,xl,logx,title) in zip(axs,((polym(1),'polynomialfeatures__degree',list(range(1,21)),'degree',False,'polynomial'),(splm(3),'splinetransformer__n_knots',list(range(3,21)),'n_knots',False,'spline'),(spl20,'ridge__alpha',list(np.logspace(-5,1,13)),'$\\alpha$',True,'20 knots + ridge'))):
    tr,va=validation_curve(model,x,y,param_name=name,param_range=vals,cv=cv,scoring='neg_root_mean_squared_error'); tr=-tr.mean(1); va=-va.mean(1); k=int(np.argmin(va))
    (ax.semilogx if logx else ax.plot)(vals,tr,'--',color=BLUE,lw=1.5,label='training'); (ax.semilogx if logx else ax.plot)(vals,va,'-',color=PUR,lw=1.6,label='validation')
    ax.plot(vals[k],va[k],'*',color=GREEN,ms=10,mec='white',zorder=6); ax.axhline(0.4,color=RED,lw=0.8,ls=':'); ax.set_ylim(0.2,1.9); grid_(ax); ax.set_xlabel(xl,fontsize=FS); ax.set_title(title,fontsize=FS-0.5,color=NAVY)
    NUM['val'][title]={'best':float(vals[k]),'cv':round(float(va[k]),3),'values':[round(float(v),3) for v in vals],'va':[round(float(v),3) for v in va]}
axs[0].set_ylabel('RMSE',fontsize=FS); axs[1].legend(fontsize=FS-0.5,loc='upper left',handlelength=2.0)
axs[0].text(11.5,0.43,'noise 0.4',fontsize=FS-0.5,color=RED)
fig.tight_layout(w_pad=0.6); save(fig,'f1592.png')

# ================= f1561 flow chart
fig,ax=plt.subplots(figsize=(4.9,2.3),dpi=240); ax.axis('off')
def fb(x0,x1,yc,t,fc=LB,ec=BLUE,fs=8.5,h=1.5): box(ax,x0,yc-h/2,x1-x0,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(0.2,4.0,4.2,'Input\n$\\mathbf{X}$ ($n\\times p$), $\\mathbf{y}$',fc=LGREY,ec=GREY)
fb(4.5,8.3,4.2,'optional:\nStandardScaler',fc='#F1EAF7',ec=PUR)
fb(8.8,13.8,4.2,'PolynomialFeatures or\nSplineTransformer:\nlearn powers or knots',fc='#FDF0E0',ec=OR)
fb(0.2,4.0,1.2,'Basis matrix\n$\\mathbf{\\Phi}$: $n$ rows,\n$m$ columns',fc='#E3F4EA',ec=GREEN)
fb(4.5,8.3,1.2,'LinearRegression\nor Ridge: weights\n$\\hat{\\mathbf{w}}$, intercept $b$',fc=LB,ec=BLUE)
fb(8.8,13.8,1.2,'Predict: same scaler,\nsame map, then\n$\\mathbf{\\Phi}\\hat{\\mathbf{w}}+b$',fc=LGREY,ec=GREY)
A((4.0,4.2),(4.5,4.2)); A((8.3,4.2),(8.8,4.2)); A((4.0,1.2),(4.5,1.2)); A((8.3,1.2),(8.8,1.2))
ax.plot([11.3,11.3],[3.45,2.7],color=NAVY,lw=0.9); ax.plot([11.3,2.1],[2.7,2.7],color=NAVY,lw=0.9); A((2.1,2.7),(2.1,1.95))
ax.set_xlim(0,14); ax.set_ylim(0.3,5.1); save(fig,'f1561.png')

# ================= f15101 case study picture
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0); j=list(d.feature_names).index('bmi')
xb=Xtr[:,[j]]; xbt=Xte[:,[j]]; g=np.linspace(xb.min(),xb.max(),200)[:,None]
fig,axs=plt.subplots(1,2,figsize=(5.0,2.4),dpi=240,gridspec_kw={'width_ratios':[1,1.1]}); ax=axs[0]
ax.scatter(xb[:,0],ytr,s=7,color=BLUE,alpha=0.55,lw=0); ax.scatter(xbt[:,0],yte,s=7,color=GREY,alpha=0.7,lw=0,marker='s')
for nm,mm,ls in (('line',LinearRegression(),'-'),('degree 3',make_pipeline(PolynomialFeatures(3,include_bias=False),LinearRegression()),'--'),('spline',splm(5),':')):
    mm.fit(xb,ytr); ax.plot(g[:,0],mm.predict(g),color=GREEN,lw=1.8,ls=ls,label=nm)
ax.legend(fontsize=FS-0.5,loc='lower right',handlelength=2.4,borderaxespad=0.2,frameon=True,framealpha=0.92,edgecolor='none',facecolor='white'); grid_(ax); ax.set_xlabel('bmi (centred, scaled)',fontsize=FS); ax.set_ylabel('progression',fontsize=FS); ax.set_title('three fits to one feature',fontsize=FS,color=NAVY)
ax=axs[1]; import re
rows=[l for l in open('verify/case_diabetes.out').read().splitlines()[2:8]]
names=[];cvs=[];tes=[]
for l in rows:
    mt=re.match(r'(.{22}) +(\d+) +([-0-9.]+) +([-0-9.]+)',l); names.append(mt.group(1).strip()); cvs.append(float(mt.group(3))); tes.append(float(mt.group(4)))
lab=['bmi: line','bmi: degree 2','bmi: degree 3','bmi: spline','ten: line','ten: deg 2 + ridge']
yy=np.arange(6)[::-1]
for yv,a_,b_ in zip(yy,cvs,tes): ax.plot([b_,a_],[yv,yv],color=LGREY,lw=2.2,zorder=1)
ax.scatter(cvs,yy,s=34,color=BLUE,zorder=3,label='CV $R^2$ (training part)'); ax.scatter(tes,yy,s=30,color=GREY,marker='s',zorder=3,label='test $R^2$')
ax.set_yticks(yy); ax.set_yticklabels(lab,fontsize=FS-0.5); ax.set_xlim(0,0.65); ax.set_ylim(-0.7,5.5); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_xlabel('$R^2$   (circle: CV, square: test)',fontsize=FS-0.5); ax.set_title('the same six models',fontsize=FS,color=NAVY)
fig.tight_layout(w_pad=1.0); save(fig,'f15101.png')

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print('figs ok')

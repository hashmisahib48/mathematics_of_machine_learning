# All figures of Chapter 9. Run: python3 fig/figs.py   (writes PNGs into ch09/ and numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Ellipse
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
NUM={}
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10]); W,Bc=1.7,0.9
def axes(ax,xl='distance $x$ (km)',yl='time $y$ (min)'):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts(ax,xx=x,yy=y,s=34,**k): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,**k)

# ---- 9.0-1 opener: a cloud, the line, the residuals
rng=np.random.default_rng(3); xo=np.sort(rng.uniform(0.5,9.5,46)); yo=1.5+0.8*xo+rng.normal(0,0.9,46)
m=LinearRegression().fit(xo[:,None],yo); po=m.predict(xo[:,None])
fig,ax=plt.subplots(figsize=(5.6,2.0),dpi=240)
for a,b_,c in zip(xo,yo,po): ax.plot([a,a],[b_,c],color=RED,lw=0.9,alpha=0.75,zorder=2)
xx=np.array([0,10]); ax.plot(xx,m.predict(xx[:,None]),color=GREEN,lw=2.2,zorder=3); pts(ax,xo,yo,s=26)
ax.text(7.2,m.predict([[7.2]])[0]-1.9,'fitted line',color=GREEN,ha='left',fontsize=10,weight='bold')
ax.annotate('residual',xy=(xo[8],(yo[8]+po[8])/2),xytext=(xo[8]-1.9,(yo[8]+po[8])/2+2.2),color=RED,fontsize=10,arrowprops=dict(arrowstyle='-',color=RED,lw=0.7))
ax.text(0.3,9.3,'data point',color=BLUE,fontsize=10); ax.set_xlim(0,10); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel('feature  $x$'); ax.set_ylabel('target  $y$'); save(fig,'f900.png')

# ---- 9.1-1 the delivery data
fig,ax=plt.subplots(figsize=(3.9,2.3),dpi=240); pts(ax,s=46); axes(ax)
for a,b_ in zip(x,y): ax.text(a+0.13,b_-0.75,f'({a:.0f}, {b_:.0f})',fontsize=8,color=GREY)
ax.axvline(6,color=PUR,lw=1,ls='--'); ax.text(6.08,3.2,'6 km:\nhow long?',color=PUR,fontsize=9)
ax.set_xlim(0,7.3); ax.set_ylim(0,13); save(fig,'f911.png')

# ---- 9.2-1 three candidate lines with squares
cands=[('Line A:  $\\hat y = 1 + 1.5x$',1.0,1.5),('Line B:  $\\hat y = 2x$',0.0,2.0),('Least squares:  $\\hat y = 0.9 + 1.7x$',0.9,1.7)]
fig,axs=plt.subplots(1,3,figsize=(6.6,2.55),dpi=240,sharey=True); NUM['cand_sse']=[]
for ax,(t,b_,w_) in zip(axs,cands):
    p=b_+w_*x; r=y-p; sse=float(np.sum(r**2)); NUM['cand_sse'].append(round(sse,2))
    for a,yy,pp,rr in zip(x,y,p,r):
        if abs(rr)>1e-9: ax.add_patch(Rectangle((a,min(yy,pp)),abs(rr),abs(rr),fc=PINK,ec=RED,lw=0.7,zorder=2))
        ax.plot([a,a],[yy,pp],color=RED,lw=1.1,zorder=3)
    xx=np.array([0,6.6]); ax.plot(xx,b_+w_*xx,color=GREEN,lw=1.8,zorder=4); pts(ax,s=26)
    ax.set_title(t,fontsize=9,color=NAVY); ax.set_aspect('equal'); ax.set_xlim(0,6.6); ax.set_ylim(0,12); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    ax.text(6.35,0.5,f'SSE = {sse:.2f}',ha='right',fontsize=10,color=RED,weight='bold'); ax.set_xlabel('$x$')
axs[0].set_ylabel('$y$'); fig.tight_layout(w_pad=0.6); save(fig,'f921.png')

# ---- 9.2-2 the loss bowl
bb=np.linspace(-3,4.8,240); ww=np.linspace(0.6,2.8,240); Bg,Wg=np.meshgrid(bb,ww)
J=((y[None,None,:]-Bg[...,None]-Wg[...,None]*x[None,None,:])**2).sum(-1)
fig,axs=plt.subplots(1,2,figsize=(6.4,2.7),dpi=240)
ax=axs[0]; cs=ax.contourf(Bg,Wg,J,levels=[1.1,2,4,8,16,32,64,128,300],cmap='Blues_r',alpha=0.9,norm=matplotlib.colors.LogNorm())
ax.contour(Bg,Wg,J,levels=[2,4,8,16,32,64,128],colors='white',linewidths=0.6)
for lab,(t,b_,w_),c,dx,dy in zip('AB*',cands,[OR,OR,GREEN],[0.35,-0.3,0.45],[-0.28,0.12,0.22]):
    ax.plot(b_,w_,'o' if lab!='*' else '*',color=c,ms=7 if lab!='*' else 13,mec='white',mew=0.8,zorder=5)
    ax.text(b_+dx,w_+dy,{'A':'A (2.75)','B':'B (2.00)','*':'minimum (1.10)'}[lab],fontsize=9,color='white' if lab!='*' else NAVY,weight='bold',ha='right' if lab=='B' else 'left',bbox=dict(fc=NAVY if lab!='*' else 'white',ec='none',pad=1.5,alpha=0.85))
ax.set_xlabel('intercept $b$'); ax.set_ylabel('slope $w$'); ax.set_title('SSE as a map over $(b, w)$',fontsize=9.5,color=NAVY)
ax=axs[1]; wl=np.linspace(0.9,2.5,200); Jl=np.array([np.sum((y-0.9-w_*x)**2) for w_ in wl]); ax.plot(wl,Jl,color=RED,lw=1.8)
ax.plot(1.7,1.1,'*',color=GREEN,ms=13,mec='white',zorder=5); ax.axvline(1.7,color=GREEN,lw=0.8,ls='--')
w0=1.15; j0=np.sum((y-0.9-w0*x)**2); g0=-2*np.sum(x*(y-0.9-w0*x)); tt=np.array([w0-0.22,w0+0.3]); ax.plot(tt,j0+g0*(tt-w0),color=PUR,lw=1.3); ax.plot(w0,j0,'o',color=PUR,ms=5)
ax.text(w0+0.08,j0+1.5,'slope of the curve\n$\\partial J/\\partial w < 0$',fontsize=8.5,color=PUR); ax.text(1.73,9,'flat here:\n$\\partial J/\\partial w = 0$',fontsize=8.5,color=GREEN)
ax.set_xlabel('slope $w$   (with $b = 0.9$)'); ax.set_ylabel('SSE'); ax.grid(color=GRID,lw=0.5); ax.set_title('A slice through the bowl',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.2); save(fig,'f922.png')

# ---- 9.3-1 anatomy of the line
fig,ax=plt.subplots(figsize=(4.6,3.0),dpi=240); xx=np.array([0,5.12]); ax.plot(xx,Bc+W*xx,color=GREEN,lw=2); pts(ax,s=40); axes(ax)
ax.plot([0],[Bc],'o',color=GREEN,ms=6,clip_on=False,zorder=7); ax.annotate('intercept $b = 0.9$\n(prediction at $x = 0$)',xy=(0.05,Bc),xytext=(1.3,0.25),fontsize=8.5,color=GREEN,va='bottom',arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.6))
ax.plot([2.4,3.4,3.4],[Bc+W*2.4]*2+[Bc+W*3.4],color=PUR,lw=1.2); ax.text(2.9,Bc+W*2.4-0.8,'run = 1',fontsize=8.5,color=PUR,ha='center'); ax.text(3.5,Bc+W*2.9-0.55,'rise = $w$ = 1.7',fontsize=8.5,color=PUR)
ax.plot([5,5],[10,9.4],color=RED,lw=1.8,zorder=4); ax.plot(5,9.4,'o',mfc='white',mec=GREEN,ms=6,zorder=6)
ax.text(5.3,10.35,'$y_5 = 10$  (measured)',fontsize=8.5,color=BLUE,va='center'); ax.text(5.3,9.0,'$\\hat y_5 = 9.4$  (predicted)',fontsize=8.5,color=GREEN,va='center')
ax.annotate('residual\n$r_5 = y_5-\\hat y_5 = 0.6$',xy=(4.97,9.7),xytext=(1.2,10.4),fontsize=8.5,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.6))
ax.set_xlim(0,8.6); ax.set_ylim(0,12.6); save(fig,'f931.png')

# ---- 9.4-1 shapes of X, w, y
fig,ax=new(6.2,2.5)
def grid(ax,x0,y0,nr,nc,c,fc,lab,dim,cw=0.46,ch=0.3,vals=None):
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x0+j*cw,y0-(i+1)*ch),cw,ch,fc=fc if not(vals and nc==2 and j==0) else LGREY,ec=c,lw=0.7))
            if vals: ax.text(x0+j*cw+cw/2,y0-(i+.5)*ch,vals[i][j],ha='center',va='center',fontsize=8,color=NAVY)
    ax.text(x0+nc*cw/2,y0+0.3,lab,ha='center',fontsize=11,color=c,weight='bold'); ax.text(x0+nc*cw/2,y0-nr*ch-0.1,dim,ha='center',va='top',fontsize=8,color=GREY)
n=5
grid(ax,0.1,2,n,1,GREEN,'#E3F4EA','$\\hat{\\mathbf{y}}$','$n$ = 5\npredictions',vals=[[v] for v in ['2.6','4.3','6.0','7.7','9.4']])
ax.text(0.86,1.25,'=',fontsize=13,ha='center',va='center')
grid(ax,1.15,2,n,2,BLUE,LB,'$\\mathbf{X}$','5 rows (samples)\n2 columns',vals=[['1',f'{v:.0f}'] for v in x])
grid(ax,2.45,2-0.45,2,1,PUR,'#F1EAF7','$\\mathbf{w}$','$b$ on top,\n$w$ below',vals=[['0.9'],['1.7']])
ax.text(1.15+0.23,2.07,'ones',fontsize=7.5,color=GREY,ha='center'); ax.text(1.15+0.69,2.07,'$x$',fontsize=8.5,color=GREY,ha='center')
ax.add_patch(Rectangle((1.15,2-3*0.3),0.92,0.3,fc='none',ec=OR,lw=1.8,zorder=6)); ax.add_patch(Rectangle((0.1,2-3*0.3),0.46,0.3,fc='none',ec=OR,lw=1.8,zorder=6))
box(ax,3.45,1.55,2.9,0.5,'row 3:   $1\\times 0.9 \\;+\\; 3\\times 1.7 \\;=\\; 6.0$',fc='#FDF0E0',ec=OR,fs=9.5)
box(ax,3.45,0.6,2.9,0.66,'every prediction is one row of $\\mathbf{X}$\nmultiplied into the same $\\mathbf{w}$',fc=LGREY,ec=GRID,fs=8.8)
ax.set_xlim(0,6.45); ax.set_ylim(-0.05,2.65); save(fig,'f941.png')

# ---- 9.4-2 projection geometry
fig,ax=new(5.4,3.0)
ax.add_patch(Polygon([(0.1,0.3),(4.3,0.3),(5.5,1.6),(1.3,1.6)],fc=LB,ec=BLUE,lw=0.8,alpha=0.85))
ax.text(0.55,0.4,'the plane of all possible predictions $\\mathbf{Xw}$',fontsize=8.5,color=BLUE,ha='left',va='bottom')
O=(1.2,0.95); Yh=(3.9,1.2); Yv=(3.9,2.9)
arr(ax,O,Yv,c=NAVY,lw=1.6,ms=10); arr(ax,O,Yh,c=GREEN,lw=1.8,ms=10); ax.plot([Yh[0],Yv[0]],[Yh[1],Yv[1]],color=RED,lw=1.6,ls=(0,(4,2)))
ax.plot([3.75,3.75,3.9],[1.19,1.34,1.355],color=RED,lw=0.8)
ax.text(2.8,2.5,'$\\mathbf{y}$  (measured)',fontsize=10,color=NAVY,ha='right'); ax.text(3.0,1.0,'$\\hat{\\mathbf{y}} = \\mathbf{X}\\hat{\\mathbf{w}}$',fontsize=10,color=GREEN,ha='center',va='top')
ax.text(4.02,2.15,'$\\mathbf{r} = \\mathbf{y}-\\hat{\\mathbf{y}}$\nshortest when\nperpendicular\nto the plane',fontsize=9,color=RED,va='center')
Zp=(2.2,1.42); arr(ax,O,Zp,c=OR,lw=1.0,ms=7); ax.plot([Zp[0],Yv[0]],[Zp[1],Yv[1]],color=OR,lw=1.0,ls=':'); ax.text(0.05,1.72,'any other $\\mathbf{Xw}$\nmisses by more',fontsize=8,color=OR,ha='left',va='bottom')
ax.set_xlim(0,5.6); ax.set_ylim(0.2,3.0); save(fig,'f942.png')

# ---- 9.6-1 flow chart of fit()
fig,ax=new(6.4,5.3)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.4): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
fb(3.5,5.55,4.7,0.45,'Input:  $\\mathbf{X}$ ($n$ rows, $p$ columns)  and  $\\mathbf{y}$ ($n$ values)',fc=LGREY,ec=GREY)
fb(3.5,4.8,4.7,0.45,'Validate: numbers only, no gaps, matching lengths')
fb(3.5,3.9,4.7,0.75,'If fit_intercept: centre the data\n$\\mathbf{X}_c=\\mathbf{X}-\\bar{\\mathbf{x}}$,   $\\mathbf{y}_c=\\mathbf{y}-\\bar y$')
ax.add_patch(Polygon([(3.5,3.4),(4.7,3.0),(3.5,2.6),(2.3,3.0)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(3.5,3.0,'which case?',ha='center',va='center',fontsize=8.6,color=NAVY)
fb(1.1,1.8,2.1,1.0,'positive=True\nnon-negative\nleast squares',fc='#F1EAF7',ec=PUR)
fb(3.5,1.8,2.3,1.0,'dense $\\mathbf{X}$ (the usual case)\nSVD-based least squares\n(Eq. 9.4-10)',fc='#E3F4EA',ec=GREEN)
fb(5.9,1.8,2.1,1.0,'sparse $\\mathbf{X}$\niterative solver\n(LSQR)',fc='#F1EAF7',ec=PUR)
fb(3.5,0.7,3.9,0.5,'Intercept:  $b=\\bar y-\\bar{\\mathbf{x}}^{\\mathrm{T}}\\hat{\\mathbf{w}}$   (zero if not fitted)')
fb(3.5,-0.05,4.9,0.45,'Store  coef_  and  intercept_ ;  return the fitted model',fc=LGREY,ec=GREY)
A((3.5,5.32),(3.5,5.03)); A((3.5,4.57),(3.5,4.27)); A((3.5,3.52),(3.5,3.41)); A((3.5,2.6),(3.5,2.31))
ax.plot([2.3,1.1],[3.0,3.0],color=NAVY,lw=0.9); A((1.1,3.0),(1.1,2.31)); ax.plot([4.7,5.9],[3.0,3.0],color=NAVY,lw=0.9); A((5.9,3.0),(5.9,2.31))
A((3.5,1.3),(3.5,0.96)); ax.plot([1.1,1.1],[1.3,0.7],color=NAVY,lw=0.9); A((1.1,0.7),(1.54,0.7)); ax.plot([5.9,5.9],[1.3,0.7],color=NAVY,lw=0.9); A((5.9,0.7),(5.46,0.7))
A((3.5,0.45),(3.5,0.19))
ax.set_xlim(0,7); ax.set_ylim(-0.32,5.85); save(fig,'f961.png')

# ---- 9.9-1 one outlier
fig,axs=plt.subplots(1,2,figsize=(6.2,2.6),dpi=240,sharey=True); y2=y.copy(); y2[4]=20; m2=LinearRegression().fit(x[:,None],y2)
NUM['outlier']={'w':round(float(m2.coef_[0]),2),'b':round(float(m2.intercept_),2),'pred6':round(float(m2.predict([[6]])[0]),2)}
for ax,yy,ww_,bb_,t in zip(axs,[y,y2],[W,m2.coef_[0]],[Bc,m2.intercept_],['Clean data','One wrong entry: 10 typed as 20']):
    xx=np.array([0,6.5]); ax.plot(xx,Bc+W*xx,color=GREEN,lw=1.0 if yy is y2 else 1.8,ls='--' if yy is y2 else '-')
    if yy is y2: ax.plot(xx,bb_+ww_*xx,color=GREEN,lw=1.8); ax.plot(5,20,'o',mfc='none',mec=RED,ms=14,mew=1.4); ax.text(4.5,20,'outlier',color=RED,fontsize=9,ha='right',va='center')
    pts(ax,x,yy); ax.set_title(t,fontsize=9.5,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_xlabel('$x$'); ax.text(0.25,22,f'$\\hat y = {bb_:.1f} {"+" if ww_>0 else ""}{ww_:.1f}x$',fontsize=10,color=GREEN)
axs[0].set_ylabel('$y$'); axs[0].set_ylim(-3,24); fig.tight_layout(w_pad=0.8); save(fig,'f991.png')

# ---- 9.9-2 collinearity
rng=np.random.default_rng(1); fig,axs=plt.subplots(1,2,figsize=(6.2,2.75),dpi=240,sharex=True,sharey=True); NUM['collin']={}
for ax,rho,t in zip(axs,[0.0,0.999],['Independent features  (correlation 0)','Nearly identical features  (correlation 0.999)']):
    cs=[]
    for k in range(300):
        z=rng.normal(size=(50,2)); x1=z[:,0]; x2=rho*z[:,0]+np.sqrt(1-rho**2)*z[:,1]; Xm=np.c_[x1,x2]; yy=x1+x2+rng.normal(0,1,50)
        cs.append(LinearRegression().fit(Xm,yy).coef_)
    cs=np.array(cs); NUM['collin'][str(rho)]={'sd_w1':round(float(cs[:,0].std()),2),'sd_w2':round(float(cs[:,1].std()),2),'sd_sum':round(float(cs.sum(1).std()),2)}
    ax.scatter(cs[:,0],cs[:,1],s=7,color=PUR,alpha=0.55,lw=0); ax.axvline(1,color=GREEN,lw=0.8,zorder=1); ax.axhline(1,color=GREEN,lw=0.8,zorder=1)
    ax.set_title(t,fontsize=9,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_xlabel('fitted $w_1$'); ax.text(0.04,0.06,f's.d. of $w_1$ = {cs[:,0].std():.2f}',transform=ax.transAxes,fontsize=8.8,color=PUR)
axs[0].set_ylabel('fitted $w_2$'); axs[0].set_xlim(-11,13); axs[0].set_ylim(-11,13); axs[0].text(1.4,11.2,'true $w_1$ = 1',color=GREEN,fontsize=8.5,va='top'); axs[0].text(-10.5,1.5,'true $w_2$ = 1',color=GREEN,fontsize=8.5)
fig.tight_layout(w_pad=0.8); save(fig,'f992.png')

# ---- 9.9-3 a line through a curve
rng=np.random.default_rng(5); xc=np.linspace(0,6,40); yc=0.5+0.45*xc**2+rng.normal(0,0.6,40); mc=LinearRegression().fit(xc[:,None],yc); rc=yc-mc.predict(xc[:,None])
NUM['curve_r2']=round(float(mc.score(xc[:,None],yc)),3)
fig,axs=plt.subplots(1,2,figsize=(6.2,2.0),dpi=240)
axs[0].plot(xc,mc.predict(xc[:,None]),color=GREEN,lw=1.8); pts(axs[0],xc,yc,s=18); axs[0].set_title(f'Fit looks acceptable:  $R^2$ = {NUM["curve_r2"]:.2f}',fontsize=9.5,color=NAVY); axs[0].set_xlabel('$x$'); axs[0].set_ylabel('$y$'); axs[0].grid(color=GRID,lw=0.5)
axs[1].axhline(0,color=GREEN,lw=1); axs[1].vlines(xc,0,rc,color=RED,lw=0.9); axs[1].scatter(xc,rc,s=16,color=RED,zorder=4); axs[1].set_title('Residuals show the missed curve',fontsize=9.5,color=NAVY)
axs[1].set_xlabel('$x$'); axs[1].set_ylabel('residual  $r = y-\\hat y$'); axs[1].grid(color=GRID,lw=0.5); fig.tight_layout(w_pad=1.0); save(fig,'f993.png')

# ---- 9.11 case study
d=load_diabetes(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0); ols=LinearRegression().fit(Xtr,ytr); p=ols.predict(Xte)
fig,axs=plt.subplots(1,2,figsize=(6.3,2.9),dpi=240)
ax=axs[0]; ax.plot([20,330],[20,330],color=GREEN,lw=1.3); ax.scatter(yte,p,s=14,color=GREY,alpha=0.8,lw=0); ax.axhline(ytr.mean(),color=OR,lw=1,ls='--'); ax.text(326,ytr.mean()-5,'mean baseline',color=OR,fontsize=8,ha='right',va='top',bbox=dict(fc='white',ec='none',pad=1,alpha=0.85))
ax.text(30,322,'green line:\nperfect prediction',color=GREEN,fontsize=8.5,va='top'); ax.set_xlabel('measured progression'); ax.set_ylabel('predicted progression'); ax.set_title('Test set: predicted against measured',fontsize=9.5,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_xlim(20,330); ax.set_ylim(20,330); ax.set_aspect('equal')
ax=axs[1]; o=np.argsort(ols.coef_); ax.barh(np.array(d.feature_names)[o],ols.coef_[o],color=[RED if v<0 else BLUE for v in ols.coef_[o]],height=0.62); ax.axvline(0,color=GREY,lw=0.7)
ax.set_title('Fitted coefficients',fontsize=9.5,color=NAVY); ax.set_xlabel('coefficient $w_j$'); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); save(fig,'f9111.png')
NUM['diab']={'corr_s1_s2':round(float(np.corrcoef(d.data[:,4],d.data[:,5])[0,1]),3)}
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

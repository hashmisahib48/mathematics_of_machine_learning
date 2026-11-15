# All figures of Chapter 23. Run: python3 fig/figs.py  (PNGs into ch23/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
plt.rcParams.update({'xtick.labelsize':8,'ytick.labelsize':8,'axes.labelsize':8.5,'legend.fontsize':8})
NUM={}; CHK={}
GL='#DDF0E4'
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grd(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# ---- delivery data
Xd=np.array([[1.],[2.],[3.],[4.],[5.]]); yd=np.array([3.,4.,6.,7.,10.])
def lsq(x,y): w,b=np.polyfit(x,y,1); return w,b
# ---- the noisy sine of Chapter 21
rng=np.random.default_rng(4); xs=np.sort(rng.uniform(0,6,28)); ys=np.sin(xs)+0.25*rng.normal(size=28); Xs=xs[:,None]
grid=np.linspace(0,6,300); G=grid[:,None]
def tube(ax,x,f,eps,col=GL): ax.fill_between(x,f-eps,f+eps,color=col,lw=0,zorder=1)
def rings(ax,x,y,idx_free,idx_bound,s=70):
    if len(idx_free): ax.scatter(x[idx_free],y[idx_free],s=s,facecolor='none',edgecolor=NAVY,lw=1.1,zorder=6)
    if len(idx_bound): ax.scatter(x[idx_bound],y[idx_bound],s=s,facecolor='none',edgecolor=RED,lw=1.1,zorder=6)
def pts(ax,x,y,s=22,c=BLUE): ax.scatter(x,y,s=s,color=c,edgecolor='white',lw=0.6,zorder=5)
def status(model,C,eps,x,y):
    f=model.predict(x[:,None]); r=y-f; d=np.zeros(len(y)); d[model.support_]=model.dual_coef_[0]
    bound=np.abs(d)>=C*(1-1e-6); free=(np.abs(d)>1e-9)&~bound
    return f,r,d,np.where(free)[0],np.where(bound)[0]

# ======== 23.0-1 opener
fig,axs=plt.subplots(1,2,figsize=(5.0,2.15),dpi=240,gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0]; m=SVR(kernel='linear',C=1,epsilon=1,tol=1e-10).fit(Xd,yd); xx=np.linspace(0.3,6.2,50); f=m.predict(xx[:,None])
tube(ax,xx,f,1.0); ax.plot(xx,f,color=GREEN,lw=2,zorder=3); w0,b0=lsq(Xd[:,0],yd); ax.plot(xx,w0*xx+b0,color=GREY,lw=1.1,ls='--',zorder=2)
pts(ax,Xd[:,0],yd,s=30); rings(ax,Xd[:,0],yd,m.support_,[],s=110); grd(ax)
ax.set_xlabel('distance  $x$ (km)'); ax.set_ylabel('time  $y$ (min)'); ax.set_xlim(0.3,6.2); ax.set_ylim(1,12.5)
ax.text(0.45,11.6,'$\\varepsilon=1$, two support vectors',fontsize=8.5,color=NAVY); ax.text(3.55,3.0,'least squares\n(dashed)',fontsize=8,color=GREY)
ax=axs[1]; m=SVR(C=10,epsilon=0.2,tol=1e-8).fit(Xs,ys); f=m.predict(G); fr,r,d,idf,idb=status(m,10,0.2,xs,ys)
tube(ax,grid,f,0.2); ax.plot(grid,np.sin(grid),color=GREY,lw=0.9,ls=':',zorder=2); ax.plot(grid,f,color=GREEN,lw=2,zorder=3)
inside=np.setdiff1d(np.arange(28),np.r_[idf,idb]); pts(ax,xs[inside],ys[inside],s=18,c='#9DB8DE'); pts(ax,xs[np.r_[idf,idb]],ys[np.r_[idf,idb]],s=26)
rings(ax,xs,ys,idf,idb,s=64); grd(ax); ax.set_xlabel('feature  $x$'); ax.set_ylim(-1.7,1.7)
ax.text(0.1,1.35,'RBF kernel, $\\varepsilon=0.2$: %d of 28 are support vectors'%len(m.support_),fontsize=8.5,color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig,'f2300.png',470)
NUM['open']={'sine_sv':int(len(m.support_)),'sine_free':int(len(idf)),'sine_bound':int(len(idb)),'sine_inside':int(len(inside)),
  'outside':int((np.abs(r)>0.2+1e-4).sum())}
# ======== 23.1-1 the wrong entry
fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240,sharey=True)
y2=yd.copy(); y2[4]=20.
xx=np.linspace(0.3,7.0,50); NUM['wrong']={}
for ax,yy,ttl,key in zip(axs,[yd,y2],['clean data','one wrong entry'],['clean','wrong']):
    m=SVR(kernel='linear',C=1,epsilon=1,tol=1e-10).fit(Xd,yy); f=m.predict(xx[:,None]); w0,b0=lsq(Xd[:,0],yy)
    tube(ax,xx,f,1.0); ax.plot(xx,f,color=GREEN,lw=2,zorder=3); ax.plot(xx,w0*xx+b0,color=GREY,lw=1.1,ls='--',zorder=2)
    pts(ax,Xd[:,0],yy,s=30); rings(ax,Xd[:,0],yy,m.support_,[],s=110) if key=='clean' else rings(ax,Xd[:,0],yy,[],m.support_,s=110)
    ax.axvline(6,color=GRID,lw=0.9,ls=':'); grd(ax)
    pl=m.predict([[6.]])[0]; pg=w0*6+b0
    ax.text(0.45,20.2,ttl,fontsize=9,color=NAVY,va='top')
    ax.text(6.12,pg+(0.0 if key=='wrong' else 0.5),'%.1f'%pg,fontsize=8.5,color=GREY,ha='left',va='center')
    ax.text(6.12,pl-(0.0 if key=='wrong' else 0.5),'%.1f'%pl,fontsize=8.5,color=GREEN,ha='left',va='center')
    ax.set_xlabel('distance  $x$ (km)'); ax.set_xlim(0.3,7.0); ax.set_ylim(0,21)
    NUM['wrong'][key]={'svr_w':round(float(m.coef_[0,0]),4),'svr_b':round(float(m.intercept_[0]),4),'svr_pred6':round(float(pl),3),'ls_w':round(float(w0),4),'ls_b':round(float(b0),4),'ls_pred6':round(float(pg),3),'sv':[int(v) for v in m.support_]}
axs[0].set_ylabel('time  $y$ (min)'); fig.tight_layout(w_pad=0.8); sv(fig,'f2311.png',470)
# ======== 23.2-1 losses and slack
fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; r=np.linspace(-3,3,300); eps=1.0
ax.axvspan(-eps,eps,color='#E8DDF2',lw=0,zorder=0)
ax.plot(r,r**2,color=GREY,lw=1.2,ls='--',label='squared'); ax.plot(r,np.abs(r),color=BLUE,lw=1.4,label='absolute'); ax.plot(r,np.maximum(0,np.abs(r)-eps),color=RED,lw=2.2,label='$\\varepsilon$-insensitive')
ax.set_ylim(-0.1,4); ax.set_xlim(-3,3); ax.set_xlabel('residual  $r=y-f(\\mathbf{x})$'); ax.set_ylabel('loss'); grd(ax)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),fontsize=7.8,handlelength=1.6,borderaxespad=0.1)
ax.text(0,0.35,'$\\varepsilon=1$',ha='center',fontsize=8.2,color=PUR)
ax=axs[1]; m=SVR(C=10,epsilon=0.2,tol=1e-8).fit(Xs,ys); f=m.predict(G); fr=m.predict(Xs); res=ys-fr
tube(ax,grid,f,0.2); ax.plot(grid,f,color=GREEN,lw=2,zorder=3); pts(ax,xs,ys,s=20)
up=int(np.argmin(abs(xs-2.25))); lo=int(np.argmin(abs(xs-0.8)))
for k,lab,sg in ((up,'$\\xi$',1),(lo,'$\\xi^{*}$',-1)):
    ax.plot([xs[k],xs[k]],[fr[k]+sg*0.2,ys[k]],color=RED,lw=2.2,zorder=4)
    ax.text(xs[k]+0.12,(fr[k]+sg*0.2+ys[k])/2,lab,color=RED,fontsize=9.5,va='center')
ax.text(4.55,0.82,'tube $f\\pm\\varepsilon$',fontsize=8.2,color=GREEN)
ax.set_xlabel('feature  $x$'); ax.set_ylim(-1.7,1.7); grd(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f2321.png',470)
NUM['slack']={'xi_up':round(float(res[up]-0.2),3),'xi_lo':round(float(-res[lo]-0.2),3)}
# ======== 23.3-1 three kinds of points; beta against residual
fig,axs=plt.subplots(1,2,figsize=(4.9,2.25),dpi=240,gridspec_kw={'width_ratios':[1.15,1]})
C3,E3=10.0,0.2
m=SVR(C=C3,epsilon=E3,tol=1e-8).fit(Xs,ys); f,res,d,idf,idb=status(m,C3,E3,xs,ys); inside=np.setdiff1d(np.arange(28),np.r_[idf,idb])
ax=axs[0]; tube(ax,grid,m.predict(G),E3); ax.plot(grid,m.predict(G),color=GREEN,lw=2,zorder=3)
pts(ax,xs[inside],ys[inside],s=20,c='#9DB8DE'); pts(ax,xs[idf],ys[idf],s=26); pts(ax,xs[idb],ys[idb],s=26); rings(ax,xs,ys,idf,idb,s=64)
ax.set_xlabel('feature  $x$'); ax.set_ylim(-1.7,1.7); grd(ax)
ax.scatter([],[],s=20,color='#9DB8DE',label='inside: $\\beta=0$'); ax.scatter([],[],s=64,facecolor='none',edgecolor=NAVY,lw=1.1,label='on the edge'); ax.scatter([],[],s=64,facecolor='none',edgecolor=RED,lw=1.1,label='outside: $|\\beta|=C$')
ax.legend(loc='upper right',fontsize=7.6,handletextpad=0.2,borderaxespad=0.1,labelspacing=0.25)
ax=axs[1]; ax.axvspan(-E3,E3,color='#E8DDF2',lw=0,zorder=0)
ax.scatter(res[inside],d[inside],s=20,color='#9DB8DE',edgecolor='white',lw=0.5,zorder=4); ax.scatter(res[idf],d[idf],s=28,color=BLUE,edgecolor=NAVY,lw=1.0,zorder=5); ax.scatter(res[idb],d[idb],s=28,color=BLUE,edgecolor=RED,lw=1.1,zorder=5)
rr=np.linspace(-0.8,0.8,400); ax.plot(rr,np.clip(np.sign(rr)*C3*(np.abs(rr)>E3+1e-12),-C3,C3),color=GREY,lw=0.8,ls=':',zorder=1)
ax.axhline(0,color=GRID,lw=0.8); ax.set_xlabel('residual  $r=y-f(x)$'); ax.set_ylabel('$\\beta=\\alpha-\\alpha^{*}$'); ax.set_ylim(-11.5,11.5); ax.set_xlim(-0.72,0.72)
ax.text(0,-4.2,'$\\pm\\varepsilon$',ha='center',fontsize=8.5,color=PUR); grd(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f2331.png',470)
NUM['kinds']={'C':C3,'eps':E3,'n':28,'inside':int(len(inside)),'free':int(len(idf)),'bound':int(len(idb)),'sv':int(len(m.support_)),
  'outside_tube':int((np.abs(res)>E3+1e-4).sum()),'max_free_res_dev':round(float(np.abs(np.abs(res[idf])-E3).max()),6),'min_bound_res':round(float(np.abs(res[idb]).min()),4),'max_inside_res':round(float(np.abs(res[inside]).max()),4),
  'sum_beta':round(float(d.sum()),8)}
# ======== 23.3-2 kernel ridge against SVR with three wrong targets
r2=np.random.default_rng(11); bad=r2.choice(28,3,replace=False); yb=ys.copy(); yb[bad]+=np.array([3.,-3.,3.])
def rm(p): return float(np.sqrt(np.mean((p-np.sin(grid))**2)))
kr={}; 
for a in (0.01,0.03,0.1,0.3,1,3,10):
    k=KernelRidge(alpha=a,kernel='rbf',gamma=0.5).fit(Xs,yb-yb.mean()); kr[a]=(rm(k.predict(G)+yb.mean()),k)
sr={}
for Cc in (0.1,0.3,1,3,10):
    s_=SVR(C=Cc,epsilon=0.2,gamma=0.5).fit(Xs,yb); sr[Cc]=(rm(s_.predict(G)),s_)
ba=min(kr,key=lambda a:kr[a][0]); bc=min(sr,key=lambda a:sr[a][0])
fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240,sharey=True)
ax=axs[0]; pk=kr[ba][1].predict(G)+yb.mean(); ax.plot(grid,np.sin(grid),color=GREY,lw=0.9,ls=':',zorder=2); ax.plot(grid,pk,color=GREEN,lw=2,zorder=3); pts(ax,xs,yb,s=20); pts(ax,xs[bad],yb[bad],s=26,c=RED)
ax.set_title('kernel ridge, $\\alpha=%g$'%ba,fontsize=9,color=NAVY); ax.set_xlabel('feature  $x$'); grd(ax)
ax=axs[1]; s_=sr[bc][1]; ps=s_.predict(G); tube(ax,grid,ps,0.2); ax.plot(grid,np.sin(grid),color=GREY,lw=0.9,ls=':',zorder=2); ax.plot(grid,ps,color=GREEN,lw=2,zorder=3); pts(ax,xs,yb,s=20); pts(ax,xs[bad],yb[bad],s=26,c=RED)
ax.scatter(xs[s_.support_],yb[s_.support_],s=60,facecolor='none',edgecolor=NAVY,lw=0.9,zorder=6)
ax.set_title('SVR, $C=%g$, $\\varepsilon=0.2$'%bc,fontsize=9,color=NAVY); ax.set_xlabel('feature  $x$'); grd(ax); ax.set_ylim(-4.3,4.6)
fig.tight_layout(w_pad=0.8); sv(fig,'f2332.png',470)
NUM['robust']={'bad_x':[round(float(v),2) for v in xs[bad]],'bad_y':[round(float(v),2) for v in yb[bad]],'krr':{str(a):round(v[0],3) for a,v in kr.items()},'svr':{str(a):round(v[0],3) for a,v in sr.items()},'best_krr':[ba,round(kr[ba][0],3)],'best_svr':[bc,round(sr[bc][0],3)],'svr_nsv':int(len(s_.support_)),
  'svr_clean':round(rm(SVR(C=bc,epsilon=0.2,gamma=0.5).fit(Xs,ys).predict(G)),3),'krr_clean':round(rm(KernelRidge(alpha=ba,kernel='rbf',gamma=0.5).fit(Xs,ys-ys.mean()).predict(G)+ys.mean()),3)}
# ======== 23.4-1 nu property
rg=np.random.default_rng(0); Xn=np.sort(rg.uniform(0,6,200))[:,None]; yn=np.sin(Xn[:,0])+0.25*rg.normal(size=200)
nus=np.linspace(0.05,0.95,19); outs=[]; shares=[]; epss=[]
for nu in nus:
    m=NuSVR(nu=nu,C=10.0,gamma=0.5,tol=1e-8).fit(Xn,yn); r=yn-m.predict(Xn); dd=np.abs(m.dual_coef_[0]); fr=(dd>1e-6)&(dd<10-1e-6)
    e=np.abs(r[m.support_[fr]]).mean(); outs.append(float(np.mean(np.abs(r)>e+1e-6))); shares.append(len(m.support_)/200); epss.append(float(e))
fig,axs=plt.subplots(1,2,figsize=(3.5,1.9),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; ax.plot([0,1],[0,1],color=GREY,lw=0.9,ls=':'); ax.fill_between(nus,outs,shares,color='#E8DDF2',lw=0); ax.plot(nus,outs,color=RED,lw=1.6,label='outside tube'); ax.plot(nus,shares,color=NAVY,lw=1.6,label='support vectors')
ax.set_xlabel('$\\nu$'); ax.set_ylabel('share of samples'); ax.set_xlim(0,1); ax.set_ylim(0,1); grd(ax); ax.legend(loc='upper left',fontsize=7.6,handlelength=1.4,borderaxespad=0.2)
ax=axs[1]; ax.plot(nus,epss,color=PUR,lw=1.6); ax.set_xlabel('$\\nu$'); ax.set_ylabel('tube half-width $\\varepsilon$'); ax.set_xlim(0,1); ax.set_ylim(0,0.6); grd(ax)
fig.tight_layout(w_pad=0.9); sv(fig,'f2341.png',336)
NUM['nu']={'nus':[round(float(v),2) for v in nus],'outs':[round(v,3) for v in outs],'shares':[round(v,3) for v in shares],'eps':[round(v,4) for v in epss]}
# ======== 23.6-1 flow chart
fig,ax=new(3.5,4.7)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=7.6,ls='-'):
    ax.add_patch(FancyBboxPatch((xc-w/2,yc-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=0.9,ls=ls)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.25)
A=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
W_=6.5; x0=3.5
fb(x0,9.05,W_,0.62,'Input:  $\\mathbf{X}$ ($n\\times p$),  $\\mathbf{y}$,  optional sample_weight',fc=LGREY,ec=GREY)
fb(x0,7.95,W_,0.78,'Validate;  turn gamma="scale" or "auto"\ninto a number (Eq. 22.5-4)')
fb(x0,6.6,W_,1.15,'Set up the dual of Eq. 23.3-5: unknowns\n$\\alpha_i,\\;\\alpha_i^{*}$ in $[0,\\;C\\times\\mathrm{weight}_i]$, linear terms\n$\\varepsilon-y_i$ and $\\varepsilon+y_i$, one equality',fc='#E3F4EA',ec=GREEN)
fb(x0,5.0,W_,1.15,'SMO of Section 22.6 on the $2n$ unknowns:\ntwo at a time, clipped to the box; kernel cache,\nshrinking; stop when the KKT gap < tol',fc='#E3F4EA',ec=GREEN)
fb(x0,3.45,W_,1.15,'$\\beta_i=\\alpha_i-\\alpha_i^{*}$;  keep the samples with $\\beta_i\\neq 0$;\nintercept from the free support vectors\n(Eq. 23.3-8)')
fb(x0,2.15,W_,0.78,'Store  support_,  support_vectors_,\ndual_coef_,  intercept_',fc=LGREY,ec=GREY)
fb(x0,0.6,W_,0.95,'NuSVR: the unknowns sum to $C\\nu n$ and $\\varepsilon$ is found.\nLinearSVR: liblinear, no kernel, no SMO.',fc='#F1EAF7',ec=PUR,ls='--')
for y1,y2 in ((8.74,8.34),(7.56,7.18),(6.02,5.58),(4.42,4.02),(2.87,2.54)): A((x0,y1),(x0,y2))
ax.set_xlim(0,7); ax.set_ylim(0.0,9.5); sv(fig,'f2361.png',336)
# ======== 23.9-1 galleries
def panel(ax,C,e,gm,title):
    m=SVR(C=C,epsilon=e,gamma=gm,tol=1e-6).fit(Xs,ys); f=m.predict(G)
    ax.fill_between(grid,f-e,f+e,color=GL,lw=0,zorder=1); ax.plot(grid,np.sin(grid),color=GREY,lw=0.7,ls=':',zorder=2); ax.plot(grid,f,color=GREEN,lw=1.5,zorder=3)
    ax.scatter(xs,ys,s=7,color=BLUE,zorder=5,lw=0); ax.scatter(xs[m.support_],ys[m.support_],s=26,facecolor='none',edgecolor=NAVY,lw=0.6,zorder=6)
    ax.set_xlim(0,6); ax.set_ylim(-1.8,1.8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    d=rm(f); ax.set_title(title+'\n%d SV, %.2f'%(len(m.support_),d),fontsize=7.6,color=NAVY,pad=2,linespacing=1.15)
    return len(m.support_),d
fig,axs=plt.subplots(3,4,figsize=(4.9,3.85),dpi=240); GAL={}
rows=[('eps',[(10,0.0,0.5),(10,0.1,0.5),(10,0.3,0.5),(10,0.8,0.5)],lambda C,e,g:'$\\varepsilon=%g$'%e),
      ('C',[(0.01,0.2,0.5),(0.1,0.2,0.5),(1,0.2,0.5),(100,0.2,0.5)],lambda C,e,g:'$C=%g$'%C),
      ('gamma',[(10,0.2,0.02),(10,0.2,0.5),(10,0.2,5),(10,0.2,100)],lambda C,e,g:'$\\gamma=%g$'%g)]
def rm(p): return float(np.sqrt(np.mean((p-np.sin(grid))**2)))
for i,(nm,cfgs,lab) in enumerate(rows):
    GAL[nm]=[]
    for j,(C,e,g) in enumerate(cfgs):
        n_,d_=panel(axs[i,j],C,e,g,lab(C,e,g)); GAL[nm].append({'C':C,'eps':e,'gamma':g,'n_sv':n_,'rmse':round(d_,3)})
fig.tight_layout(h_pad=0.5,w_pad=0.3); sv(fig,'f2391.png',470)
NUM['gallery']=GAL
# ======== 23.9-2 validation heat map and curve
Cs=10.0**np.arange(-1,2.51,0.5); Es=np.round(np.arange(0,0.81,0.1),2); Gs=10.0**np.arange(-2,1.01,0.5)
folds=list(KFold(5,shuffle=True,random_state=0).split(xs))
def cvr(C,e,g):
    err=[];tr=[];nsv=[]
    for itr,ite in folds:
        m=SVR(C=C,epsilon=e,gamma=g,tol=1e-4).fit(Xs[itr],ys[itr]); err.append(np.mean((m.predict(Xs[ite])-ys[ite])**2)); tr.append(np.mean((m.predict(Xs[itr])-ys[itr])**2)); nsv.append(len(m.support_)/len(itr))
    return float(np.sqrt(np.mean(err))),float(np.sqrt(np.mean(tr))),float(np.mean(nsv))
CV=np.array([[[cvr(C,e,g)[0] for g in Gs] for e in Es] for C in Cs]); ic,ie,ig=np.unravel_index(CV.argmin(),CV.shape)
Cb,Eb,Gb=float(Cs[ic]),float(Es[ie]),float(Gs[ig])
NUM['cv']={'best_C':Cb,'best_eps':Eb,'best_gamma':Gb,'best_rmse':round(float(CV.min()),4),'noise_sd':0.25}
m=SVR(C=Cb,epsilon=Eb,gamma=Gb).fit(Xs,ys); NUM['cv']['full_fit_rmse_truth']=round(rm(m.predict(G)),3); NUM['cv']['full_fit_nsv']=int(len(m.support_))
mk=KernelRidge(alpha=0.032,kernel='rbf',gamma=0.1).fit(Xs,ys-ys.mean()); NUM['cv']['krr_ch21_rmse_truth']=round(rm(mk.predict(G)+ys.mean()),3)
fig,axs=plt.subplots(1,2,figsize=(4.9,2.35),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; H_=CV[:,:,ig]
im=ax.pcolormesh(Es,np.log10(Cs),np.log10(np.minimum(H_,1.0)),cmap='Blues_r',shading='nearest',vmin=np.log10(0.25),vmax=np.log10(1.0))
ax.plot(Eb,np.log10(Cb),'*',color=OR,ms=11,mec='white',mew=0.7); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('$\\log_{10}C$'); ax.set_title('Cross-validated RMSE, $\\gamma=%.2g$'%Gb,fontsize=8.5,color=NAVY)
cb=fig.colorbar(im,ax=ax,pad=0.03,fraction=0.05); cb.set_ticks(np.log10([0.25,0.5,1.0])); cb.set_ticklabels(['0.25','0.5','≥ 1']); cb.ax.tick_params(labelsize=8)
ax=axs[1]; rr=np.array([cvr(Cb,e,Gb) for e in Es])
ax.plot(Es,rr[:,1],color=BLUE,lw=1.5,marker='o',ms=3,label='training folds'); ax.plot(Es,rr[:,0],color=GREY,lw=1.5,marker='s',ms=3,label='validation folds')
ax.axvline(Eb,color=PUR,lw=1,ls='--'); ax.axhline(0.25,color=RED,lw=0.8,ls=':'); ax.text(0.8,0.238,'noise level 0.25',color=RED,fontsize=8,ha='right',va='top')
ax.set_ylim(0,0.62); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('RMSE'); grd(ax); ax.legend(loc='upper center',bbox_to_anchor=(0.52,1.0),fontsize=7.8,handlelength=1.4,borderaxespad=0.2)
ax2=ax.twinx(); ax2.plot(Es,rr[:,2],color=NAVY,lw=1.0,ls=(0,(3,2))); ax2.set_ylim(0,1.05); ax2.set_ylabel('share of SVs',fontsize=8.5,color=NAVY); ax2.tick_params(labelsize=8,colors=NAVY); ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(NAVY)
ax.set_title('Validation curve in $\\varepsilon$, $C=%.2g$'%Cb,fontsize=8.5,color=NAVY)
fig.tight_layout(w_pad=0.7); sv(fig,'f2392.png',470)
NUM['cv']['curve']={'eps':[float(v) for v in Es],'train':[round(float(v),3) for v in rr[:,1]],'valid':[round(float(v),3) for v in rr[:,0]],'sv_share':[round(float(v),3) for v in rr[:,2]]}
# ======== 23.10-1 three noise densities
Cc,Ee=2.0,0.5; r=np.linspace(-3,3,2001)
pe=Cc/(2*(1+Cc*Ee))*np.exp(-Cc*np.maximum(0,np.abs(r)-Ee)); pl=Cc/2*np.exp(-Cc*np.abs(r))
var_e=float(np.trapezoid(pe*r**2,r)); pg=np.exp(-r**2/(2*var_e))/np.sqrt(2*np.pi*var_e)
fig,ax=plt.subplots(figsize=(3.3,1.9),dpi=240)
ax.axvspan(-Ee,Ee,color='#E8DDF2',lw=0,zorder=0); ax.plot(r,pg,color=GREY,lw=1.2,ls='--',label='normal (least squares)'); ax.plot(r,pl,color=BLUE,lw=1.4,label='Laplace ($\\varepsilon=0$)'); ax.plot(r,pe,color=RED,lw=2.0,label='$\\varepsilon$-insensitive')
ax.set_xlim(-3,3); ax.set_ylim(0,1.2); ax.set_xlabel('error  $r$'); ax.set_ylabel('density'); grd(ax); ax.legend(loc='upper right',fontsize=7.6,handlelength=1.5,borderaxespad=0.2)
fig.tight_layout(); sv(fig,'f23101.png',316)
NUM['density']={'C':Cc,'eps':Ee,'flat_height':round(Cc/(2*(1+Cc*Ee)),4),'integral':round(float(np.trapezoid(Cc/(2*(1+Cc*Ee))*np.exp(-Cc*np.maximum(0,np.abs(np.linspace(-40,40,400001))-Ee)),np.linspace(-40,40,400001))),5),'var_eps':round(var_e,4),'sd_eps':round(var_e**0.5,4),'laplace_peak':round(Cc/2,4)}
# ======== 23.11-1 case study
Xdb,ydb=load_diabetes(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xdb,ydb,test_size=0.25,random_state=0)
def scaled(est): return TransformedTargetRegressor(make_pipeline(StandardScaler(),est),transformer=StandardScaler())
g_=10**-1.5
mods={'OLS':LinearRegression(),'ridge':RidgeCV(alphas=np.logspace(-3,1,41)),'KRR rbf':scaled(KernelRidge(alpha=1,kernel='rbf',gamma=g_)),'SVR linear':scaled(SVR(kernel='linear',C=g_,epsilon=0.4)),'SVR rbf':scaled(SVR(C=1,epsilon=0.3,gamma=g_))}
nb=int(0.15*len(ytr)); sc={k:[] for k in mods}
for seed in range(5):
    rg=np.random.default_rng(seed); bad=rg.choice(len(ytr),nb,replace=False); yw=ytr.copy(); yw[bad]+=rg.choice([-300.,300.],nb)
    for k,mm in mods.items(): sc[k].append(mm.fit(Xtr,yw).score(Xte,yte))
clean={k:mm.fit(Xtr,ytr).score(Xte,yte) for k,mm in mods.items()}
NUM['case']={k:{'clean':round(float(clean[k]),3),'mean':round(float(np.mean(sc[k])),3),'min':round(float(np.min(sc[k])),3),'max':round(float(np.max(sc[k])),3)} for k in mods}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.35),dpi=240,gridspec_kw={'width_ratios':[1,1.2]})
ax=axs[0]; ax.plot([0,350],[0,350],color=GRID,lw=1)
mm=mods['SVR linear'].fit(Xtr,ytr); p=mm.predict(Xte); ax.scatter(yte,p,s=14,color=GREEN,alpha=0.8,lw=0)
ax.text(30,305,'SVR linear, test $R^2=%.2f$'%mm.score(Xte,yte),fontsize=8,color=NAVY)
ax.set_xlabel('measured progression'); ax.set_ylabel('predicted'); ax.set_xlim(20,350); ax.set_ylim(40,330); grd(ax)
ax=axs[1]; ks=list(mods); xx=np.arange(len(ks)); wd=0.36
ax.bar(xx-wd/2,[clean[k] for k in ks],wd,color='#B9C9E3',label='clean training')
ax.bar(xx+wd/2,[np.mean(sc[k]) for k in ks],wd,color=[GREY,GREY,GREY,GREEN,GREEN],label='15 % wrecked (mean)')
for i,k in enumerate(ks): ax.plot([i+wd/2]*2,[min(sc[k]),max(sc[k])],color=NAVY,lw=1.2); ax.text(i+wd/2,max(sc[k])+0.02,'%.2f'%np.mean(sc[k]),ha='center',fontsize=7.6,color=NAVY)
ax.set_xticks(xx); ax.set_xticklabels(['OLS','ridge','KRR\nrbf','SVR\nlinear','SVR\nrbf'],fontsize=8); ax.set_ylim(0,0.6); ax.set_ylabel('test $R^2$'); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#B9C9E3',label='clean training'),Patch(color=GREY,label='wrecked: mean, range')],loc='upper center',fontsize=7.6,borderaxespad=0.1,handlelength=1.2,ncol=2,columnspacing=0.8,handletextpad=0.4)
fig.tight_layout(w_pad=0.9); sv(fig,'f23111.png',470)
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)

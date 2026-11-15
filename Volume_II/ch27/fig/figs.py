# All figures of Chapter 27. Run: python3 fig/figs.py  (PNGs into ch27/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from scipy.special import expit
from sklearn.ensemble import GradientBoostingRegressor as GR, GradientBoostingClassifier as GC
from sklearn.datasets import make_moons, make_friedman1, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
LO='#FDEBD0'; CMAP=ListedColormap([LB,LO])
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def stepfun(ax,xs,ys,**k): ax.plot(xs,ys,drawstyle='steps-post',**k)

# ---------------------------------------------------------------- the toy: six lemonade days
xt=np.array([1,2,3,4,5,6.]); yt=np.array([3,5,4,10,14,12.]); XT=xt[:,None]
gt=GR(n_estimators=3,max_depth=1,learning_rate=0.5,random_state=0).fit(XT,yt)
xg=np.linspace(0.5,6.5,241); XG=xg[:,None]
stg=[np.full_like(xg,gt.init_.constant_.item())]+[p for p in gt.staged_predict(XG)]
NUM['toy']={'F0':float(gt.init_.constant_.item()),'train_mse':[round(float(v),4) for v in gt.train_score_],
            'thresholds':[float(e[0].tree_.threshold[0]) for e in gt.estimators_],
            'leaves':[[round(float(v),4) for v in e[0].tree_.value[1:3,0,0]] for e in gt.estimators_]}

# ---- 27.0-1 opener: model after 0, 3, 20, 100 trees on a noisy sine
rng=np.random.RandomState(3); xs=np.sort(rng.uniform(0,6,40)); ys=np.sin(xs)+rng.normal(0,0.3,40)
xx=np.linspace(0,6,300)[:,None]
gs=GR(n_estimators=100,max_depth=1,learning_rate=0.1,random_state=0).fit(xs[:,None],ys)
stp=list(gs.staged_predict(xx)); c0=gs.init_.constant_.item()
NUM['sine']={'train_mse':[round(float(np.mean((ys-c0)**2)),3)]+[round(float(gs.train_score_[k-1]),3) for k in (3,20,100)],'true_mse':[round(float(np.mean((np.full(300,c0)-np.sin(xx[:,0]))**2)),3)]+[round(float(np.mean((stp[k-1]-np.sin(xx[:,0]))**2)),3) for k in (3,20,100)]}
fig,axs=plt.subplots(1,4,figsize=(5.3,1.85),dpi=240,sharey=True)
for ax,m in zip(axs,[0,3,20,100]):
    ax.plot(xx[:,0],np.sin(xx[:,0]),color=GREY,lw=0.9,ls=':'); ax.scatter(xs,ys,s=9,color=BLUE,edgecolor='white',lw=0.4,zorder=4)
    ax.plot(xx[:,0],np.full(300,gs.init_.constant_.item()) if m==0 else stp[m-1],color=GREEN,lw=1.6)
    ax.set_title('no tree' if m==0 else f'{m} trees',fontsize=9,color=NAVY); ax.set_xticks([0,3,6]); ax.set_ylim(-1.7,1.7); ax.set_xlabel('$x$',fontsize=9)
axs[0].set_ylabel('$y$',fontsize=9); fig.tight_layout(w_pad=0.3); sv(fig,'f270.png',470)

# ---- 27.1-1 the data
fig,ax=plt.subplots(figsize=(3.4,2.4),dpi=240)
ax.scatter(xt,yt,s=42,color=BLUE,edgecolor='white',lw=0.8,zorder=5,label='day on file')
ax.axhline(8,color=GREEN,lw=1.2,ls='--'); ax.text(6.45,8.35,'mean 8',color=GREEN,fontsize=8.5,ha='right',va='bottom')
ax.scatter([3.8],[0],s=75,marker='*',color=PUR,edgecolor='white',lw=0.6,zorder=6,clip_on=False); ax.text(4.05,0.0,'new day, 3.8 hours: ?',color=PUR,fontsize=8.5,va='center')
for a,b in zip(xt,yt): ax.text(a,b+0.8,f'{b:.0f}',fontsize=8.5,color=GREY,ha='center')
ax.set_xlim(0.4,6.6); ax.set_ylim(-1,16.5); ax.set_xlabel('hours of sun'); ax.set_ylabel('glasses sold (dozens)'); grid(ax); sv(fig,'f271.png',330)

# ---- 27.2-1 three rounds on the toy
fig,axs=plt.subplots(2,3,figsize=(5.4,3.55),dpi=240,sharex=True)
for m in (1,2,3):
    ax=axs[0,m-1]; Fp=stg[m-1]; Fn=stg[m]
    prevF=(np.full(6,8.0) if m==1 else list(gt.staged_predict(XT))[m-2])
    for a,b,pf in zip(xt,yt,prevF): ax.plot([a,a],[b,pf],color=RED,lw=1.3,zorder=3)
    ax.plot(xg,Fp,color=GREY,lw=1.2,ls='--',zorder=2); ax.plot(xg,Fn,color=GREEN,lw=1.8,zorder=4)
    ax.scatter(xt,yt,s=26,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.set_ylim(1,16); ax.set_title(f'round {m}',fontsize=9,color=NAVY)
    ax.text(0.97,0.05,f'MSE {gt.train_score_[m-1]:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom')
    ax=axs[1,m-1]; resid=yt-prevF; t=gt.estimators_[m-1,0].tree_; s=t.threshold[0]; vl,vr=t.value[1,0,0],t.value[2,0,0]
    ax.axhline(0,color=GRID,lw=0.8); ax.plot([0.5,s],[vl,vl],color=GREEN,lw=1.8); ax.plot([s,6.5],[vr,vr],color=GREEN,lw=1.8); ax.axvline(s,color=PUR,lw=1.0,ls='--')
    ax.scatter(xt,resid,s=26,color=RED,marker='D',edgecolor='white',lw=0.7,zorder=5)
    ax.set_ylim(-7.5,7.5); ax.set_xlabel('hours of sun',fontsize=9); ax.set_xlim(0.4,6.6)
    ax.set_title(f'tree leaves {vl:+.2f} | {vr:+.2f}'.replace('-','\u2212'),fontsize=8.5,color=GREEN,pad=3)
axs[0,0].set_ylabel('glasses (dozens)',fontsize=9); axs[1,0].set_ylabel('residual $r$',fontsize=9)
fig.tight_layout(h_pad=0.5,w_pad=0.4); sv(fig,'f272.png',470)
NUM['round3']={'split':float(gt.estimators_[2,0].tree_.threshold[0]),'leaves':[round(float(v),4) for v in gt.estimators_[2,0].tree_.value[1:3,0,0]],'F3':[round(float(v),4) for v in gt.predict(XT)]}

# ---- 27.4-1 regression losses and their negative gradients
e=np.linspace(-3,3,601); dl=1.5; al=0.9
hub=np.where(abs(e)<=dl,0.5*e**2,dl*(abs(e)-dl/2)); pin=np.where(e>0,al*e,(al-1)*e)
fig,axs=plt.subplots(1,2,figsize=(5.3,2.35),dpi=240)
ax=axs[0]
ax.plot(e,0.5*e**2,color=BLUE,lw=1.6); ax.plot(e,abs(e),color=GREEN,lw=1.6,ls='--'); ax.plot(e,hub,color=PUR,lw=1.8,ls='-.'); ax.plot(e,pin,color=RED,lw=1.6,ls=':')
ax.set_ylim(0,3.2); ax.set_xlim(-3,3); ax.set_xlabel('residual $e = y - F$'); ax.set_ylabel('loss $\\ell$'); grid(ax)
for c_,ls_,lab_ in ((BLUE,'-','squared'),(GREEN,'--','absolute'),(PUR,'-.','Huber, $\\delta=1.5$'),(RED,':','pinball, $\\alpha=0.9$')): ax.plot([],[],color=c_,ls=ls_,lw=1.6,label=lab_)
ax.legend(loc='upper center',fontsize=8.3,borderaxespad=0.1,handlelength=2.4,labelspacing=0.25)
ax=axs[1]
ax.plot(e,e,color=BLUE,lw=1.6); ax.plot(e,np.sign(e),color=GREEN,lw=1.6,ls='--'); ax.plot(e,np.clip(e,-dl,dl),color=PUR,lw=1.8,ls='-.'); ax.plot(e,np.where(e>0,al,al-1),color=RED,lw=1.6,ls=':')
ax.set_xlim(-3,3); ax.set_ylim(-3.1,3.1); ax.axhline(0,color=GREY,lw=0.6); ax.set_xlabel('residual $e = y - F$'); ax.set_ylabel('pseudo-residual $r$'); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f274.png',470)

# ---- 27.5-1 log loss: gradient and the Newton step in a leaf
Fv=np.linspace(-6,6,400)
fig,axs=plt.subplots(1,2,figsize=(5.3,2.35),dpi=240,gridspec_kw={'width_ratios':[1,1.1]})
ax=axs[0]; ax.plot(Fv,1-expit(Fv),color=OR,lw=1.7); ax.plot(Fv,-expit(Fv),color=BLUE,lw=1.7,ls='--'); ax.axhline(0,color=GREY,lw=0.6)
ax.set_xlabel('score $F$ (log-odds)'); ax.set_ylabel('pseudo-residual $r = y - p$'); ax.set_ylim(-1.08,1.08); grid(ax)
ax.text(0.2,0.78,'$y=1$',color=OR,fontsize=9,ha='left'); ax.text(-0.2,-0.78,'$y=0$',color=BLUE,fontsize=9,ha='right')
ax=axs[1]; gam=np.linspace(-1.5,3.5,300); yv=np.array([1,1,1,1,1,0.]); p0=0.5
Lg=np.array([np.mean(np.log1p(np.exp(g))-yv*g) for g in gam]); gr=np.mean(p0-yv); hs=p0*(1-p0)
quad=np.log(2)+gr*gam+0.5*hs*gam**2; gN=-gr/hs; gstar=np.log(5)
NUM['leaf']={'newton':round(float(gN),4),'exact':round(float(gstar),4)}
ax.plot(gam,Lg,color=RED,lw=1.8,label='mean log loss'); ax.plot(gam,quad,color=PUR,lw=1.3,ls='--',label='parabola at $\\gamma=0$')
ax.scatter([gN],[np.log(2)+gr*gN+0.5*hs*gN**2],s=40,color=PUR,zorder=5,marker='o',label='Newton step 1.33'); ax.scatter([gstar],[np.mean(np.log1p(np.exp(gstar))-yv*gstar)],s=60,color=GREEN,zorder=5,marker='*',label='minimum 1.61')
ax.legend(loc='lower right',fontsize=8.0,borderaxespad=0.1,handlelength=1.8,labelspacing=0.25)
ax.axvline(0,color=GREY,lw=0.6); ax.set_xlabel('leaf value $\\gamma$'); ax.set_ylabel('loss of the leaf'); ax.set_ylim(0.0,1.0); ax.set_xlim(-1.5,3.5); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f275.png',470)

# ---- 27.8-1 flow chart of fit
fig,ax=new(3.9,5.6)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
def Ln(*pts): ax.plot([p[0] for p in pts],[p[1] for p in pts],color=NAVY,lw=0.9)
def dia(xc,yc,w,h,t,fs=8.2): ax.add_patch(Polygon([(xc,yc+h/2),(xc+w/2,yc),(xc,yc-h/2),(xc-w/2,yc)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.1)
X0=2.55
fb(X0,9.1,4.6,0.6,'Input:  $\\mathbf{X}$,  $y$,  loss $\\ell$,  $\\eta$,  $M$',fc=LGREY,ec=GREY)
fb(X0,8.2,4.6,0.6,'$F_0$ = best constant (init);\n$\\mathbf{F}$ = $F_0$ for every sample',fc=LB,ec=BLUE)
fb(X0,7.1,4.6,0.7,'pseudo-residuals\n$r_i = -\\partial\\ell(y_i,F)/\\partial F$  at the current $\\mathbf{F}$')
fb(X0,5.95,4.6,0.7,'draw the in-bag samples\n(subsample < 1; else all)')
fb(X0,4.8,4.6,0.7,'fit a regression tree to $r$\n(max_depth, max_features, ...)')
fb(X0,3.65,4.6,0.7,'line search in each leaf:\n$\\gamma_t$ = median, quantile or Newton step')
fb(X0,2.5,4.6,0.7,'$\\mathbf{F} \\leftarrow \\mathbf{F} + \\eta\\,\\gamma_{t(i)}$;  store tree and loss',fc='#E3F4EA',ec=GREEN)
dia(X0,1.2,4.2,1.2,'$m < M$ and validation\nloss still falling?')
fb(X0,-0.15,4.6,0.6,'store the trees and losses;  return self',fc=LGREY,ec=GREY)
for y0,y1 in [(8.8,8.5),(7.9,7.45),(6.75,6.3),(5.6,5.15),(4.45,4.0),(3.3,2.85),(2.15,1.8),(0.6,0.15)]: A((X0,y0),(X0,y1))
Ln((X0+2.1,1.2),(5.05,1.2),(5.05,7.1)); A((5.05,7.1),(X0+2.3,7.1))
ax.text(X0+0.12,0.38,'no',fontsize=8,color=GREY,style='italic',va='center'); ax.text(X0+2.2,1.45,'yes',fontsize=8,color=GREY,style='italic',va='bottom')
ax.text(5.15,4.15,'next stage\n$m \\leftarrow m+1$\n($K$ trees if\nmulticlass)',fontsize=8,color=GREY,va='center',ha='left')
ax.set_xlim(0,6.9); ax.set_ylim(-0.6,9.55); sv(fig,'f277.png',360)


def cls_pts(ax,X,y,s=30,**k):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.5,zorder=5,**k)

# ---- 27.11-1 depth gallery on the moons (the data of Figure 24.10-1)
Xm,ym=make_moons(n_samples=300,noise=0.3,random_state=0); Xmt,Xme,ymt,yme=train_test_split(Xm,ym,test_size=0.33,random_state=0,stratify=ym)
gx,gy=np.meshgrid(np.linspace(-1.7,2.7,300),np.linspace(-1.2,1.7,300)); NUM['gallery']={}
fig,axs=plt.subplots(1,4,figsize=(5.3,1.95),dpi=240,sharey=True)
for ax,d in zip(axs,[1,2,3,6]):
    m=GC(max_depth=d,n_estimators=100,random_state=0).fit(Xmt,ymt); Z=m.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx,gy,Z,levels=[0.5],colors=[GREEN],linewidths=1.3)
    cls_pts(ax,Xmt,ymt,s=7); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'depth {d}',fontsize=9,color=NAVY)
    acc=m.score(Xme,yme); ax.text(0.97,0.04,f'test {acc:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    NUM['gallery'][str(d)]={'train':round(float(m.score(Xmt,ymt)),4),'test':round(float(acc),4),'logloss_test':round(float(log_loss(yme,m.predict_proba(Xme))),4)}
fig.tight_layout(w_pad=0.3); sv(fig,'f2711a.png',470)

# ---- 27.11-2 learning rate, number of stages and subsampling on make_friedman1
Xf,yf=make_friedman1(n_samples=600,noise=1.0,random_state=0); Xft,Xfe,yft,yfe=train_test_split(Xf,yf,test_size=0.5,random_state=0)
fig,axs=plt.subplots(1,2,figsize=(5.3,2.4),dpi=240)
ax=axs[0]; NUM['eta']={}
for lr,col,ls in ((1.0,RED,':'),(0.3,OR,'-.'),(0.1,GREEN,'-'),(0.03,BLUE,'--')):
    g=GR(learning_rate=lr,n_estimators=500,max_depth=3,random_state=0).fit(Xft,yft); st=np.array([mean_squared_error(yfe,p) for p in g.staged_predict(Xfe)])
    ax.plot(np.arange(1,501),st,color=col,ls=ls,lw=1.5,label=f'$\\eta={lr}$'); ax.scatter([st.argmin()+1],[st.min()],s=26,color=col,zorder=5,edgecolor='white',lw=0.5)
    NUM['eta'][str(lr)]={'best_stage':int(st.argmin()+1),'best_mse':round(float(st.min()),3),'final_mse':round(float(st[-1]),3)}
ax.set_ylim(2.5,12); ax.set_xlabel('number of trees $M$'); ax.set_ylabel('test mean squared error'); ax.legend(loc='center right',bbox_to_anchor=(1.0,0.45),fontsize=8.3,borderaxespad=0.1,handlelength=2.2,labelspacing=0.25); grid(ax)
ax=axs[1]; NUM['subsample']={}
for ss,col,ls in ((1.0,GREEN,'-'),(0.5,PUR,'--')):
    g=GR(learning_rate=0.1,n_estimators=500,max_depth=3,subsample=ss,random_state=0).fit(Xft,yft); st=np.array([mean_squared_error(yfe,p) for p in g.staged_predict(Xfe)])
    ax.plot(np.arange(1,501),st,color=col,ls=ls,lw=1.5,label=f'test, subsample {ss}')
    NUM['subsample'][str(ss)]={'best_stage':int(st.argmin()+1),'best_mse':round(float(st.min()),3),'final_mse':round(float(st[-1]),3)}
    if ss<1:
        ax.plot(np.arange(1,501),g.oob_scores_,color=RED,ls=':',lw=1.5,label='out-of-bag loss'); NUM['subsample']['oob_last']=round(float(g.oob_scores_[-1]),4); NUM['subsample']['oob_first']=round(float(g.oob_scores_[0]),3)
ax.set_ylim(0,9); ax.set_xlabel('number of trees $M$'); ax.set_ylabel('mean squared error'); ax.legend(loc='upper right',fontsize=8.3,borderaxespad=0.1,handlelength=2.2,labelspacing=0.25); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f2711b.png',470)

# ---- 27.11-3 validation curves (5-fold, 300 training points)
fig,axs=plt.subplots(1,2,figsize=(5.3,2.35),dpi=240); NUM['valcurve']={}
for ax,(name,grid_,xs_,xl) in zip(axs,[('max_depth',[1,2,3,4,6,8],[1,2,3,4,6,8],'max_depth'),('learning_rate',[0.01,0.03,0.1,0.3,1.0],[0.01,0.03,0.1,0.3,1.0],'learning_rate')]):
    tr,va=validation_curve(GR(n_estimators=100,max_depth=3,random_state=0),Xft,yft,param_name=name,param_range=grid_,cv=5,scoring='neg_mean_squared_error')
    tr=-tr; va=-va; mt,mv,sv_=tr.mean(1),va.mean(1),va.std(1); k=int(np.argmin(mv))
    ax.plot(xs_,mt,'-o',color=BLUE,ms=3.2,lw=1.4,label='training folds'); ax.plot(xs_,mv,'-s',color=GREY,ms=3.2,lw=1.4,label='validation folds')
    ax.fill_between(xs_,mv-sv_,mv+sv_,color=GREY,alpha=0.18,lw=0); ax.axvline(xs_[k],color=PUR,lw=1.0,ls='--'); ax.set_xlabel(xl); grid(ax)
    if name=='learning_rate': ax.set_xscale('log'); ax.set_xticks(xs_); ax.set_xticklabels(['0.01','0.03','0.1','0.3','1'])
    else: ax.set_xticks(xs_)
    NUM['valcurve'][name]={'grid':grid_,'train':[round(float(v),3) for v in mt],'valid':[round(float(v),3) for v in mv],'sd':[round(float(v),3) for v in sv_],'best':xs_[k]}
axs[0].set_ylabel('mean squared error'); axs[0].set_ylim(0,14); axs[1].set_ylim(0,14); axs[0].legend(loc='upper right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
fig.tight_layout(w_pad=1.0); sv(fig,'f2711c.png',450)

# ---- 27.13-1 case study: diabetes, breast cancer
Xd,yd=load_diabetes(return_X_y=True); Xdt,Xde,ydt,yde=train_test_split(Xd,yd,test_size=0.25,random_state=0)
ols=LinearRegression().fit(Xdt,ydt); mse_ols=mean_squared_error(yde,ols.predict(Xde))
Xc,yc=load_breast_cancer(return_X_y=True); Xct,Xce,yct,yce=train_test_split(Xc,yc,test_size=0.25,random_state=0,stratify=yc)
lr_=make_pipeline(StandardScaler(),LogisticRegression()).fit(Xct,yct); ll_lr=log_loss(yce,lr_.predict_proba(Xce))
fig,axs=plt.subplots(1,3,figsize=(5.5,2.3),dpi=240,gridspec_kw={'width_ratios':[1,1,0.95]}); NUM['case']={}
ax=axs[0]
for d,col,ls in ((1,GREEN,'-'),(3,RED,'--')):
    g=GR(max_depth=d,learning_rate=0.1,n_estimators=300,random_state=0).fit(Xdt,ydt); st=np.array([mean_squared_error(yde,p) for p in g.staged_predict(Xde)])
    ax.plot(np.arange(1,301),st,color=col,ls=ls,lw=1.5,label=f'depth {d}'); NUM['case'][f'diab_depth{d}']={'best_stage':int(st.argmin()+1),'best_mse':round(float(st.min()),1),'final_mse':round(float(st[-1]),1)}
ax.axhline(mse_ols,color=GREY,lw=1.0,ls=':'); ax.text(298,mse_ols+40,'least squares',fontsize=8.3,color=GREY,ha='right',va='bottom'); NUM['case']['diab_ols_mse']=round(float(mse_ols),1)
ax.set_ylim(2900,5600); ax.set_xlabel('trees'); ax.set_ylabel('test MSE (diabetes)'); ax.legend(loc='upper right',fontsize=8.3,borderaxespad=0.1,handlelength=1.8,labelspacing=0.25); grid(ax)
ax=axs[1]
for d,col,ls in ((1,GREEN,'-'),(3,RED,'--')):
    g=GC(max_depth=d,learning_rate=0.1,n_estimators=300,random_state=0).fit(Xct,yct); st=np.array([log_loss(yce,np.c_[1-expit(f),expit(f)]) for f in g.staged_decision_function(Xce)])
    ax.plot(np.arange(1,301),st,color=col,ls=ls,lw=1.5,label=f'depth {d}'); NUM['case'][f'cancer_depth{d}']={'best_stage':int(st.argmin()+1),'best_ll':round(float(st.min()),4),'final_ll':round(float(st[-1]),4)}
ax.axhline(ll_lr,color=GREY,lw=1.0,ls=':'); ax.text(298,ll_lr-0.01,'logistic regression',fontsize=8.3,color=GREY,ha='right',va='top'); NUM['case']['cancer_lr_ll']=round(float(ll_lr),4)
ax.set_ylim(0,0.5); ax.set_xlabel('trees'); ax.set_ylabel('test log loss (cancer)'); ax.legend(loc='upper right',fontsize=8.3,borderaxespad=0.1,handlelength=1.8,labelspacing=0.25); grid(ax)
ax=axs[2]; g=GC(random_state=0).fit(Xct,yct); names=load_breast_cancer().feature_names; o=np.argsort(g.feature_importances_)[::-1][:5]
ax.barh(range(5),g.feature_importances_[o][::-1],color=GREEN,height=0.6); ax.set_yticks(range(5)); ax.set_yticklabels([names[k].replace('worst ','w. ').replace('mean ','m. ') for k in o][::-1],fontsize=8.3); ax.set_xlabel('importance'); ax.set_xlim(0,0.62); grid(ax)
NUM['case']['importances']=[(str(names[k]),round(float(g.feature_importances_[k]),4)) for k in o]
fig.tight_layout(w_pad=0.7); sv(fig,'f2713.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print(json.dumps(NUM)); print({k:v['min_font_printed'] for k,v in CHK.items()})

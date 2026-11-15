# All figures of Chapter 26. Run: python3 fig/figs.py  (PNGs into ch26/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.ensemble import AdaBoostClassifier as ABC, AdaBoostRegressor as ABR
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
LO='#FDEBD0'; CMAP=ListedColormap([LB,LO])
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def pts2(ax,X,y,s=30,**k):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s if np.isscalar(s) else s[m],color=col,marker=mk,edgecolor='white',lw=0.6,zorder=5,**k)
def weights_used(m,X,y):
    """sample weights at the start of every round, rebuilt from the stored stumps (K = 2 or more)"""
    n=len(y); w=np.full(n,1/n); out=[]; cls=m.classes_
    for h,a in zip(m.estimators_,m.estimator_weights_):
        out.append(w.copy()); w=w*np.exp(a*(h.predict(X)!=y)); w/=w.sum()
    return out

# ---------------------------------------------------------------- the six apples
x6=np.arange(1,7.); y6=np.array([-1,-1,-1,1,-1,1]); X6=x6[:,None]
ada6=ABC(n_estimators=3,random_state=0).fit(X6,y6); W6=weights_used(ada6,X6,y6)
def stump_fn(h,xx): return h.predict(xx[:,None])
xx=np.linspace(0.4,6.6,400); Fx=sum(a*stump_fn(h,xx) for h,a in zip(ada6.estimators_,ada6.estimator_weights_))
NUM['toy']={'alpha':[round(float(a),4) for a in ada6.estimator_weights_],'F_at_x':[round(float(v),3) for v in sum(a*stump_fn(h,x6) for h,a in zip(ada6.estimators_,ada6.estimator_weights_))]}

# ---- 26.0-1 opener: three rounds on the apples, and their sum
fig,axs=plt.subplots(1,4,figsize=(5.5,2.05),dpi=240,gridspec_kw={'width_ratios':[1,1,1,1.12]})
for k,ax in enumerate(axs[:3]):
    h=ada6.estimators_[k]; p=stump_fn(h,xx); w=W6[k]
    ax.fill_between(xx,-1,1,where=p<0,color=LB,lw=0,step=None); ax.fill_between(xx,-1,1,where=p>0,color=LO,lw=0)
    t=h.tree_.threshold[0]; ax.axvline(t,color=GREEN,lw=1.6)
    for xi,yi,wi in zip(x6,y6,w):
        ax.scatter(xi,0,s=380*wi+18,marker='o' if yi<0 else 's',color=BLUE if yi<0 else OR,edgecolor='white',lw=0.6,zorder=5)
        if h.predict([[xi]])[0]!=yi: ax.scatter(xi,0,s=380*wi+18+70,facecolors='none',edgecolors=RED,lw=1.2,zorder=6)
    ax.set_xlim(0.4,6.6); ax.set_ylim(-1,1.15); ax.set_yticks([]); ax.set_xticks(range(1,7)); ax.spines['left'].set_visible(False)
    ax.set_title(f'round {k+1}',fontsize=9,color=NAVY,pad=3)
    ax.text(0.5,0.95,f'err {ada6.estimator_errors_[k]:.3f}\n$\\alpha$ = {ada6.estimator_weights_[k]:.3f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='center',va='top',linespacing=1.15,bbox=dict(fc='white',ec='none',pad=1.0,alpha=0.8),zorder=8)
ax=axs[3]; ax.axhline(0,color=GREY,lw=0.8); ax.step(xx,Fx,where='mid',color=GREEN,lw=1.8)
Fv=sum(a*stump_fn(h,x6) for h,a in zip(ada6.estimators_,ada6.estimator_weights_))
for xi,yi,fi in zip(x6,y6,Fv): ax.scatter(xi,fi,s=34,marker='o' if yi<0 else 's',color=BLUE if yi<0 else OR,edgecolor='white',lw=0.6,zorder=5)
ax.set_xlim(0.4,6.6); ax.set_xticks(range(1,7)); ax.set_ylim(-3.2,3.2); ax.set_title('sum $F(x)$',fontsize=9,color=NAVY,pad=3)
ax.set_yticks([-2,0,2])
fig.text(0.5,0.0,'sugar reading $x$     (marker area = weight of the apple in that round; red ring = misclassified by that stump)',ha='center',va='bottom',fontsize=8.5,color=GREY)
fig.tight_layout(w_pad=0.4,rect=(0,0.07,1,1)); sv(fig,'f260.png',470)

# ---- 26.2-1 intuition in the plane: weights grow on the mistakes
Xp,yp=make_moons(n_samples=40,noise=0.2,random_state=1)
adap=ABC(n_estimators=20,random_state=0).fit(Xp,yp); Wp=weights_used(adap,Xp,yp)
gx,gy=np.meshgrid(np.linspace(-1.5,2.5,250),np.linspace(-1,1.5,250)); G=np.c_[gx.ravel(),gy.ravel()]
fig,axs=plt.subplots(1,4,figsize=(5.5,1.75),dpi=240,sharey=True); NUM['plane']={}
for k,ax in enumerate(axs):
    if k<3:
        h=adap.estimators_[k]; Z=h.predict(G).reshape(gx.shape); w=Wp[k]; ttl=f'round {k+1}'
        err=adap.estimator_errors_[k]; NUM['plane'][f'err{k+1}']=round(float(err),4)
        sz=1300*w+6
    else:
        Z=adap.predict(G).reshape(gx.shape); sz=14; ttl='20 rounds'; w=None
        NUM['plane']['train20']=round(float(adap.score(Xp,yp)),4)
    ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx,gy,Z,levels=[0.5],colors=[GREEN],linewidths=1.3)
    pts2(ax,Xp,yp,s=sz)
    if k<3:
        wrong=adap.estimators_[k].predict(Xp)!=yp
        ax.scatter(Xp[wrong,0],Xp[wrong,1],s=1300*w[wrong]+6+45,facecolors='none',edgecolors=RED,lw=1.0,zorder=6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=9,color=NAVY,pad=3)
fig.tight_layout(w_pad=0.3); sv(fig,'f262.png',470)
NUM['plane']['max_w_share']=[round(float(w.max()),3) for w in Wp[:3]]

# ---- 26.3-1 exponential loss and alpha against the error
fig,axs=plt.subplots(1,2,figsize=(5.3,2.3),dpi=240,gridspec_kw={'width_ratios':[1.05,1]})
ax=axs[0]; m=np.linspace(-2,2,400)
ax.plot(m,(m<0).astype(float),color=GREY,lw=1.4,ls=':'); ax.plot(m,np.maximum(0,1-m),color=PUR,lw=1.4,ls='-.')
ax.plot(m,np.log2(1+np.exp(-m)),color=BLUE,lw=1.5,ls='--'); ax.plot(m,np.exp(-m),color=RED,lw=1.9)
ax.set_ylim(0,3.6); ax.set_xlim(-2,2); ax.set_xlabel('margin $y\\,F(\\mathbf{x})$'); ax.set_ylabel('loss'); grid(ax)
ax.legend(handles=[plt.Line2D([],[],color=RED,lw=1.9,label='exponential $e^{-yF}$'),plt.Line2D([],[],color=PUR,lw=1.4,ls='-.',label='hinge'),plt.Line2D([],[],color=BLUE,lw=1.5,ls='--',label='logistic'),plt.Line2D([],[],color=GREY,lw=1.4,ls=':',label='0–1')],loc='upper right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
ax=axs[1]; e=np.linspace(0.02,0.98,500)
for K,ls,lab in ((2,'-','K = 2'),(3,'--','K = 3'),(10,':','K = 10')):
    a_=np.log((1-e)/e)+np.log(K-1); ax.plot(e,a_,color=GREEN,lw=1.6,ls=ls,label=lab); ax.plot([1-1/K],[0],'o',color=GREEN,ms=4.5,zorder=5)
ax.axhline(0,color=GREY,lw=0.8); ax.set_xlim(0,1); ax.set_ylim(-3.5,6); ax.set_xlabel('weighted error $\\varepsilon_m$'); ax.set_ylabel('weight $\\alpha_m$  ($\\eta=1$)'); grid(ax)
ax.plot([],[],'o',color=GREEN,ms=4.5,ls='none',label='$\\varepsilon=1-1/K$'); ax.legend(loc='upper right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
fig.tight_layout(w_pad=1.0); sv(fig,'f263.png',470)

# ---- 26.4-1 AdaBoost.R2 on a noisy sine
fig,ax=plt.subplots(figsize=(3.6,2.3),dpi=240)
rng=np.random.default_rng(4); xs=np.sort(rng.uniform(0,6,80)); ys=np.sin(xs)+rng.normal(0,0.3,80); xg=np.linspace(0,6,400)[:,None]
tr=DTR(max_depth=3,random_state=0).fit(xs[:,None],ys); rg=ABR(DTR(max_depth=3),n_estimators=50,random_state=0).fit(xs[:,None],ys)
xt=np.sort(rng.uniform(0,6,400)); yt=np.sin(xt)+rng.normal(0,0.3,400)
NUM['sine']={'tree_r2_test':round(float(tr.score(xt[:,None],yt)),3),'ada_r2_test':round(float(rg.score(xt[:,None],yt)),3),'tree_r2_train':round(float(tr.score(xs[:,None],ys)),3),'ada_r2_train':round(float(rg.score(xs[:,None],ys)),3),'kept':len(rg.estimators_)}
ax.scatter(xs,ys,s=7,color=BLUE,lw=0,zorder=4); ax.plot(xg[:,0],np.sin(xg[:,0]),color=GREY,lw=0.9,ls=':')
ax.plot(xg[:,0],tr.predict(xg),color=OR,lw=1.3,ls='--',label='one tree, depth 3'); ax.plot(xg[:,0],rg.predict(xg),color=GREEN,lw=1.6,label='AdaBoost.R2, 50 trees')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_ylim(-1.9,2.5); ax.legend(loc='upper right',fontsize=8.2,borderaxespad=0.1,handlelength=1.8)
sv(fig,'f264.png',340)

# ---- 26.6-1 flow chart
fig,ax=new(5.6,2.9)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.8): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
def dia(xc,yc,w,h,t,fs=8.8): ax.add_patch(Polygon([(xc,yc+h/2),(xc+w/2,yc),(xc,yc-h/2),(xc-w/2,yc)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.1)
Ar=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
def yn(x,y,t,ha='left'): ax.text(x,y,t,fontsize=8.8,color=GREY,style='italic',ha=ha,va='center')
yT,yB=2.45,0.75
fb(0.95,yT,1.7,0.9,'$w_i=1/n$\n$m=1$',fc=LGREY,ec=GREY)
fb(3.25,yT,2.3,0.9,'fit stump $h_m$\nwith weights $w_i$')
fb(5.95,yT,2.5,0.9,'weighted error\n$\\varepsilon_m$  (Eq. 26.3-1)')
dia(9.0,yT,2.7,1.4,'$\\varepsilon_m=0$  or\n$\\varepsilon_m\\geq 1-1/K$ ?')
fb(9.0,yB,2.7,0.9,'weight $\\alpha_m$\n(Eq. 26.3-2)')
fb(6.0,yB,2.8,0.9,'update the weights\n(Eq. 26.3-3)')
dia(3.25,yB,2.3,1.2,'$m=M$ ?')
fb(0.8,yB,1.5,0.9,'return\n$h_m,\\alpha_m$',fc='#E3F4EA',ec=GREEN)
fb(9.0,3.95,2.7,0.6,'stop; keep what exists',fc='#E3F4EA',ec=GREEN,fs=8.6)
Ar((1.8,yT),(2.1,yT)); Ar((4.4,yT),(4.7,yT)); Ar((7.2,yT),(7.65,yT))
Ar((9.0,yT+0.7),(9.0,3.65)); yn(9.12,3.2,'yes')
Ar((9.0,yT-0.7),(9.0,yB+0.45)); yn(9.12,1.6,'no')
Ar((7.65,yB),(7.4,yB)); Ar((4.6,yB),(4.4,yB)); Ar((2.1,yB),(1.6,yB)); yn(1.85,yB+0.25,'yes',ha='center')
Ar((3.25,yB+0.6),(3.25,yT-0.45)); yn(3.38,1.6,'no:  $m\\leftarrow m+1$')
ax.set_xlim(-0.1,10.5); ax.set_ylim(0.1,4.35); sv(fig,'f266.png',470)

# ---- 26.9-1 gallery: the number of rounds
Xm,ym=make_moons(n_samples=300,noise=0.3,random_state=0); Xmt,Xme,ymt,yme=train_test_split(Xm,ym,test_size=0.33,random_state=0,stratify=ym)
gx2,gy2=np.meshgrid(np.linspace(-1.7,2.7,300),np.linspace(-1.2,1.7,300)); G2=np.c_[gx2.ravel(),gy2.ravel()]; NUM['gallery']={}
fig,axs=plt.subplots(1,4,figsize=(5.5,1.95),dpi=240,sharey=True)
for ax,M in zip(axs,[1,3,10,100]):
    m=ABC(n_estimators=M,random_state=0).fit(Xmt,ymt); Z=m.predict(G2).reshape(gx2.shape)
    ax.contourf(gx2,gy2,Z,levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx2,gy2,Z,levels=[0.5],colors=[GREEN],linewidths=1.3)
    pts2(ax,Xmt,ymt,s=7); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{M} stump'+('s' if M>1 else ''),fontsize=9,color=NAVY,pad=3)
    acc=m.score(Xme,yme); ax.text(0.97,0.04,f'test {acc:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    NUM['gallery'][str(M)]={'train':round(float(m.score(Xmt,ymt)),4),'test':round(float(acc),4)}
fig.tight_layout(w_pad=0.3); sv(fig,'f269a.png',470)

# ---- 26.9-2 validation curves (numbers from code/choose_ada.py)
V=json.load(open('verify/choose_ada.json'))
fig,axs=plt.subplots(1,2,figsize=(5.2,2.3),dpi=240)
ax=axs[0]; rr=np.arange(1,301)
for eta,col,ls in (('1.0',BLUE,'-'),('0.3',GREEN,'--'),('0.1',PUR,'-.')):
    v=np.array(V['eta'][eta]); ax.plot(rr,v,color=col,lw=1.4,ls=ls,label=f'$\\eta$ = {eta}')
ax.set_xscale('log'); ax.set_xlabel('rounds $M$'); ax.set_ylabel('validation accuracy'); ax.set_xticks([1,3,10,30,100,300]); ax.set_xticklabels(['1','3','10','30','100','300']); ax.set_ylim(0.70,0.95); grid(ax)
ax.legend(loc='lower right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
ax=axs[1]; gd=[1,2,3,4,6,8]; trn=[V['depth'][str(d)][0] for d in gd]; va=np.array([V['depth'][str(d)][1] for d in gd]); sd=np.array([V['depth'][str(d)][2] for d in gd])
ax.plot(gd,trn,'-o',color=BLUE,ms=3.4,lw=1.4,label='training folds'); ax.plot(gd,va,'-s',color=GREY,ms=3.4,lw=1.4,label='validation folds'); ax.fill_between(gd,va-sd,va+sd,color=GREY,alpha=0.18,lw=0)
ax.set_xlabel('depth of the base tree'); ax.set_ylabel('accuracy'); ax.set_ylim(0.78,1.02); ax.set_xticks(gd); grid(ax); ax.legend(loc='lower right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
fig.tight_layout(w_pad=1.0); sv(fig,'f269b.png',450)

# ---- 26.9-3 label noise (numbers from code/noise.py)
Nz=json.load(open('verify/noise.json')); rr=np.arange(1,401)
fig,axs=plt.subplots(1,3,figsize=(5.4,2.05),dpi=240)
for ax,d,ttl in ((axs[0],1,'stumps'),(axs[1],3,'trees of depth 3')):
    ax.plot(rr,Nz[f'{d}_clean'],color=BLUE,lw=1.5,label='clean labels'); ax.plot(rr,Nz[f'{d}_noisy'],color=RED,lw=1.5,ls='--',label='10% flipped')
    ax.set_xscale('log'); ax.set_xticks([1,10,100,400]); ax.set_xticklabels(['1','10','100','400']); ax.set_ylim(0.08,0.24); ax.set_xlabel('rounds $m$'); ax.set_title(ttl,fontsize=9,color=NAVY,pad=3); grid(ax)
axs[0].set_ylabel('test error'); axs[0].legend(loc='upper right',fontsize=8.2,borderaxespad=0.2,handlelength=1.6); axs[1].set_yticklabels([])
ax=axs[2]; ax.plot(rr,Nz['share_1'],color=GREEN,lw=1.5,label='stumps'); ax.plot(rr,Nz['share_3'],color=PUR,lw=1.5,ls='--',label='depth 3')
ax.axhline(0.10,color=GREY,lw=0.9,ls=':',label='share of points')
ax.set_xscale('log'); ax.set_xticks([1,10,100,400]); ax.set_xticklabels(['1','10','100','400']); ax.set_ylim(0,0.5); ax.set_yticks([0,0.1,0.2,0.3]); ax.set_xlabel('rounds $m$'); ax.set_ylabel('weight on flipped'); ax.set_title('where the weight goes',fontsize=9,color=NAVY,pad=3); grid(ax); ax.legend(loc='upper right',fontsize=8.0,borderaxespad=0.2,handlelength=1.6)
fig.tight_layout(w_pad=0.6); sv(fig,'f269c.png',470)

# ---- 26.10-1 case study
cd=load_breast_cancer(); Xt,Xe,yt,ye=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
a200=ABC(n_estimators=200,random_state=0).fit(Xt,yt); tr_=np.array(list(a200.staged_score(Xt,yt))); te_=np.array(list(a200.staged_score(Xe,ye)))
lr=make_pipeline(StandardScaler(),LogisticRegression()).fit(Xt,yt); lr_te=lr.score(Xe,ye)
fig,axs=plt.subplots(1,2,figsize=(5.4,2.4),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; rr=np.arange(1,201); ax.plot(rr,tr_,color=BLUE,lw=1.4,label='training'); ax.plot(rr,te_,color=OR,lw=1.4,ls='--',label='test'); ax.axhline(lr_te,color=GREEN,lw=1.0,ls=':',label='logistic, test')
ax.set_xscale('log'); ax.set_xticks([1,3,10,30,100]); ax.set_xticklabels(['1','3','10','30','100']); ax.set_ylim(0.86,1.02); ax.set_xlabel('rounds $m$'); ax.set_ylabel('accuracy'); grid(ax); ax.legend(loc='lower right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
ax=axs[1]; imp=a200.feature_importances_; top=np.argsort(imp)[::-1][:8][::-1]
ax.barh(range(8),imp[top],color=GREEN,height=0.65); ax.set_yticks(range(8)); ax.set_yticklabels([cd.feature_names[j] for j in top],fontsize=8.2); ax.set_xlabel('importance'); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f2610.png',470)
NUM['case']={'lr_test':round(float(lr_te),4),'top':[(cd.feature_names[j],round(float(imp[j]),4)) for j in top[::-1]]}
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
bad={k:v for k,v in CHK.items() if v['min_font_printed']<7.5}; print('figures below 7.5 pt:',bad); print(json.dumps(NUM))

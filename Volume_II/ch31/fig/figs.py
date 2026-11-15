# All figures of Chapter 31. Run: python3 fig/figs.py [names]  (PNGs into ch31/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from figstyle import *
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_moons, load_digits
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore',ConvergenceWarning)
ONLY=set(sys.argv[1:]); NF='verify/fig_numbers.json'
NUM=json.load(open(NF)) if os.path.exists(NF) else {}
def want(n): return (not ONLY) or n in ONLY
def fontchk(fig,name,printed_in):
    fw=fig.get_size_inches()[0]; sm=99; who=''
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip():
            fs=t.get_fontsize()*printed_in/fw
            if fs<sm: sm=fs; who=t.get_text()[:25]
    print(f'  {name}: smallest printed text {sm:.1f} pt ({who!r})'); NUM.setdefault('minfont',{})[name]=round(sm,1)
def done(fig,name,printed_in):
    fontchk(fig,name,printed_in); save(fig,name); json.dump(NUM,open(NF,'w'),indent=1)
LBC=ListedColormap(['#E3EEFB','#FDEBD2'])
# the XOR data of the chapter: x1*x2 > 0 accepted (class 0), x1*x2 < 0 rejected (class 1)
X8=np.array([[-2,-1],[-1,-2],[1,2],[2,1],[-2,1],[-1,2],[1,-2],[2,-1]],float); y8=np.array([0,0,0,0,1,1,1,1])
def parts(ax,s=46,lab=True):
    ax.scatter(X8[y8==0,0],X8[y8==0,1],s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,marker='o',label='accept')
    ax.scatter(X8[y8==1,0],X8[y8==1,1],s=s,color=OR,edgecolor='white',lw=0.8,zorder=5,marker='s',label='reject')
    if lab:
        for i,(a,b_) in enumerate(X8): ax.text(a+0.13,b_-0.38,str(i+1),fontsize=8.5,color=GREY,zorder=6)
def xyaxes(ax,lim=2.9):
    ax.axhline(0,color=GRID,lw=0.8,zorder=0); ax.axvline(0,color=GRID,lw=0.8,zorder=0)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_xlabel('width deviation $x_1$'); ax.set_ylabel('height deviation $x_2$'); ax.set_aspect('equal')
    ax.set_xticks([-2,-1,0,1,2]); ax.set_yticks([-2,-1,0,1,2])
moonsX,moonsy=make_moons(300,noise=0.3,random_state=0)
Xm_tr,Xm_te,ym_tr,ym_te=train_test_split(moonsX,moonsy,test_size=0.25,random_state=0)   # 225 / 75 as in Chapter 20
def grid2(lim=(-2.2,3.3,-1.6,2.2),n=250):
    gx,gy=np.meshgrid(np.linspace(lim[0],lim[1],n),np.linspace(lim[2],lim[3],n)); return gx,gy,np.c_[gx.ravel(),gy.ravel()]

# ---- 31.0-1 opener: network and boundary
if want('f310.png'):
    Xo,yo=make_moons(160,noise=0.22,random_state=1); Xo_te,yo_te=make_moons(1000,noise=0.22,random_state=11)
    lr=LogisticRegression().fit(Xo,yo)
    net=make_pipeline(StandardScaler(),MLPClassifier((8,),solver='lbfgs',alpha=0.01,max_iter=1000,random_state=0)).fit(Xo,yo)
    NUM['opener']={'logreg_test':round(lr.score(Xo_te,yo_te),3),'mlp_test':round(net.score(Xo_te,yo_te),3)}
    fig,axs=plt.subplots(1,2,figsize=(5.0,2.35),dpi=240,gridspec_kw={'width_ratios':[1.2,1.0]})
    ax=axs[0]; ax.axis('off'); ax.set_xlim(-0.2,4.2); ax.set_ylim(-0.3,4.7); ax.set_aspect('equal')
    xs=[0.5,2.0,3.5]; ly=[[1.6,2.8],[0.4+0.8*k for k in range(4)],[2.2]]; ly[1]=[0.35,1.35,2.35,3.35]; ly[0]=[1.25,2.45]; ly[2]=[1.85]
    for j,(ya,yb) in enumerate(zip(ly[0],[None,None])): pass
    for a in ly[0]:
        for b_ in ly[1]: ax.plot([xs[0],xs[1]],[a,b_],color=GRID,lw=0.8,zorder=1)
    for a in ly[1]: ax.plot([xs[1],xs[2]],[a,ly[2][0]],color=GRID,lw=0.8,zorder=1)
    for x_,ys,c,lab in [(xs[0],ly[0],BLUE,['$x_1$','$x_2$']),(xs[1],ly[1],GREEN,['']*4),(xs[2],ly[2],OR,['$\\hat{y}$'])]:
        for y_,l in zip(ys,lab): ax.add_patch(Circle((x_,y_),0.28,fc='white',ec=c,lw=1.6,zorder=3)); ax.text(x_,y_,l,ha='center',va='center',fontsize=10,color=NAVY,zorder=4)
    for x_,t,c in [(xs[0],'input',BLUE),(xs[1],'hidden\n4 ReLU',GREEN),(xs[2],'output\nlogistic',OR)]: ax.text(x_,4.65,t,ha='center',va='top',fontsize=8.5,color=c,weight='bold')
    ax.text(2.0,-0.25,'$W^{(1)}$: 2×4,   $W^{(2)}$: 4×1',ha='center',va='top',fontsize=8.5,color=GREY)
    ax=axs[1]; gx,gy,G=grid2((-1.9,2.9,-1.4,1.9),220)
    Z=net.predict_proba(G)[:,1].reshape(gx.shape); ax.contourf(gx,gy,Z,levels=[0,0.5,1],cmap=LBC,zorder=0); ax.contour(gx,gy,Z,levels=[0.5],colors=GREEN,linewidths=1.8,zorder=2)
    ww=lr.coef_[0]; xl=np.array([-1.9,2.9]); ax.plot(xl,-(lr.intercept_[0]+ww[0]*xl)/ww[1],color=GREY,lw=1.3,ls='--',zorder=3)
    ax.scatter(Xo[yo==0,0],Xo[yo==0,1],s=14,color=BLUE,edgecolor='white',lw=0.5,zorder=5); ax.scatter(Xo[yo==1,0],Xo[yo==1,1],s=14,color=OR,marker='s',edgecolor='white',lw=0.5,zorder=5)
    ax.text(2.85,-1.3,f'network: {NUM["opener"]["mlp_test"]:.2f}',color=GREEN,fontsize=9,ha='right',weight='bold'); ax.text(-1.85,1.75,f'straight line: {NUM["opener"]["logreg_test"]:.2f}',color=GREY,fontsize=9,ha='left',weight='bold',va='top')
    ax.set_xlim(-1.9,2.9); ax.set_ylim(-1.4,1.9); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$')
    for s in ax.spines.values(): s.set_visible(True)
    fig.tight_layout(w_pad=0.5); done(fig,'f310.png',4.9)

# ---- 31.1-1 the parts data and the best straight line
if want('f311.png'):
    best=0; bl=None; rng=np.random.default_rng(0)
    for th in np.linspace(0,np.pi,361):
        for c0 in np.linspace(-2.5,2.5,101):
            n_=np.array([np.cos(th),np.sin(th)]); sc_=X8@n_-c0; acc=max(((sc_>0)==y8).mean(),((sc_<=0)==y8).mean())
            if acc>best+1e-9: best=acc; bl=(th,c0)
    lr=LogisticRegression(C=1e6).fit(X8,y8); NUM['xor']={'best_line_acc':float(best),'logreg_acc':float(lr.score(X8,y8)),'logreg_coef':[round(float(v),6) for v in lr.coef_[0]]}
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.45),dpi=240)
    ax=axs[0]; 
    for sx,sy in [(1,1),(-1,-1)]: ax.add_patch(plt.Rectangle((0 if sx>0 else -3,0 if sy>0 else -3),3,3,fc='#E3EEFB',ec='none',zorder=0))
    for sx,sy in [(1,-1),(-1,1)]: ax.add_patch(plt.Rectangle((0 if sx>0 else -3,0 if sy>0 else -3),3,3,fc='#FDEBD2',ec='none',zorder=0))
    parts(ax); xyaxes(ax); ax.set_title('the eight parts',fontsize=9.5,color=NAVY); ax.legend(fontsize=8.5,loc='upper center',ncol=2,handletextpad=0.2,columnspacing=0.8,bbox_to_anchor=(0.5,-0.2),frameon=False)
    ax=axs[1]; parts(ax,lab=False); xyaxes(ax)
    th,c0=bl; n_=np.array([np.cos(th),np.sin(th)]); t=np.linspace(-3.5,3.5,50); p0=c0*n_; d=np.array([-n_[1],n_[0]])
    ax.plot(p0[0]+t*d[0],p0[1]+t*d[1],color=GREEN,lw=1.8,zorder=3); ax.set_title('the best straight line',fontsize=9.5,color=NAVY)
    ax.text(2.8,-2.8,f'{int(round(best*8))} of 8 correct',fontsize=9,color=GREEN,ha='right',weight='bold',va='bottom'); ax.set_ylabel('')
    fig.tight_layout(w_pad=0.6); done(fig,'f311.png',4.8)

# ---- 31.2-1 learned features: hidden plane
if want('f321.png'):
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.45),dpi=240)
    ax=axs[0]; parts(ax,lab=False); xyaxes(ax)
    ax.add_patch(FancyArrowPatch((0,0),(1.5,-1.5),arrowstyle='-|>',mutation_scale=9,color=PUR,lw=1.6,zorder=4)); ax.add_patch(FancyArrowPatch((0,0),(-1.5,1.5),arrowstyle='-|>',mutation_scale=9,color=GREEN,lw=1.6,zorder=4))
    ax.text(1.62,-1.55,'$h_1$ grows',color=PUR,fontsize=8.5,ha='left',va='center'); ax.text(-1.62,1.55,'$h_2$ grows',color=GREEN,fontsize=8.5,ha='right',va='center')
    ax.set_title('input plane',fontsize=9.5,color=NAVY)
    ax=axs[1]; h1=np.maximum(0,X8[:,0]-X8[:,1]); h2=np.maximum(0,X8[:,1]-X8[:,0])
    ax.scatter(h1[y8==0],h2[y8==0],s=46,color=BLUE,edgecolor='white',lw=0.8,zorder=5,marker='o'); ax.scatter(h1[y8==1],h2[y8==1],s=46,color=OR,edgecolor='white',lw=0.8,zorder=5,marker='s')
    for (a,b_) in {(a,b_) for a,b_ in zip(h1,h2)}: ax.text(a+0.17,b_+0.1,'×2',fontsize=8.5,color=GREY,ha='left',va='bottom')
    tt=np.linspace(0,3.6,10); ax.plot(tt,2-tt,color=GREEN,lw=1.8,zorder=3); ax.fill_between(tt,2-tt,4,color='#FDEBD2',zorder=0,lw=0); ax.fill_between(tt,0,2-tt,color='#E3EEFB',zorder=0,lw=0)
    ax.text(2.55,1.0,'$h_1 + h_2 = 2$',fontsize=8.5,color=GREEN,ha='left',va='bottom',rotation=0)
    ax.set_xlim(-0.2,3.6); ax.set_ylim(-0.2,3.6); ax.set_aspect('equal'); ax.set_xlabel('hidden value $h_1 = \\max(0, x_1 - x_2)$'); ax.set_ylabel('$h_2 = \\max(0, x_2 - x_1)$'); ax.set_xticks([0,1,2,3]); ax.set_yticks([0,1,2,3]); ax.grid(color=GRID,lw=0.5)
    ax.text(0.2,0.35,'accept',fontsize=8.5,color=BLUE,ha='left',va='center'); ax.text(2.3,2.4,'reject',fontsize=8.5,color='#B9770E',ha='right',va='center')
    ax.set_title('hidden plane',fontsize=9.5,color=NAVY)
    NUM['hidden']={'h_sum_accept':sorted(set((h1+h2)[y8==0].tolist())),'h_sum_reject':sorted(set((h1+h2)[y8==1].tolist()))}
    fig.tight_layout(w_pad=0.6); done(fig,'f321.png',4.8)

# ---- 31.2-2 a curve from ReLU units
if want('f322.png'):
    rng=np.random.default_rng(2); xs=np.sort(rng.uniform(0,2*np.pi,30)); ys=np.sin(xs)+rng.normal(0,0.1,30)
    best=None
    for s in range(30):
        m=MLPRegressor(hidden_layer_sizes=(5,),solver='lbfgs',alpha=1e-3,max_iter=3000,random_state=s).fit(xs[:,None],ys)
        if best is None or m.loss_<best.loss_: best=m
    m=best; xg=np.linspace(0,2*np.pi,400); W1=m.coefs_[0][0]; b1=m.intercepts_[0]; W2=m.coefs_[1][:,0]; b2=m.intercepts_[1][0]
    contrib=np.array([W2[j]*np.maximum(0,W1[j]*xg+b1[j]) for j in range(5)]); tot=b2+contrib.sum(0); assert np.allclose(tot,m.predict(xg[:,None]))
    kinks=[-b1[j]/W1[j] for j in range(5) if W1[j]!=0 and 0<-b1[j]/W1[j]<2*np.pi]
    NUM['relu_curve']={'train_mse':round(float(((m.predict(xs[:,None])-ys)**2).mean()),4),'kinks':[round(float(k),2) for k in kinks],'n_active_kinks':len(kinks)}
    cols=[BLUE,OR,GREEN,PUR,RED]
    fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240,sharey=False)
    ax=axs[0]
    for j in range(5): ax.plot(xg,contrib[j],color=cols[j],lw=1.5,label=f'unit {j+1}')
    ax.axhline(0,color=GRID,lw=0.8); ax.set_title('five ReLU units, each scaled by its output weight',fontsize=8.5,color=NAVY); ax.set_xlabel('$x$'); ax.set_ylabel('contribution'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8,ncol=1,loc='upper right',handlelength=1.2,labelspacing=0.2)
    ax=axs[1]; ax.scatter(xs,ys,s=14,color=BLUE,edgecolor='white',lw=0.5,zorder=4); ax.plot(xg,np.sin(xg),color=GREY,lw=1,ls='--',label='$\\sin x$'); ax.plot(xg,tot,color=GREEN,lw=2,label='their sum + $b$')
    for k in kinks: ax.plot([k],[b2+sum(W2[j]*max(0,W1[j]*k+b1[j]) for j in range(5))],'o',color=PUR,ms=4,zorder=5)
    ax.set_title('the sum is piecewise linear',fontsize=8.5,color=NAVY); ax.set_xlabel('$x$'); ax.set_ylabel('$\\hat{y}$'); ax.grid(color=GRID,lw=0.5); ax.legend(fontsize=8,loc='lower left',handlelength=1.5,labelspacing=0.2)
    fig.tight_layout(w_pad=0.7); done(fig,'f322.png',4.8)

# ---- 31.3-1 activations and derivatives
if want('f331.png'):
    z=np.linspace(-4,4,801); sg=1/(1+np.exp(-z))
    F=[('identity',z,np.ones_like(z),'$z$','$1$'),('logistic',sg,sg*(1-sg),'$1/(1+e^{-z})$','$a(1-a)$'),('tanh',np.tanh(z),1-np.tanh(z)**2,'$\\tanh z$','$1-a^2$'),('relu',np.maximum(z,0),(z>0)*1.0,'$\\max(0,z)$','1 if $z>0$, else 0')]
    fig,axs=plt.subplots(2,2,figsize=(4.9,3.5),dpi=240,sharex=True)
    for ax,(nm,f,df,fl,dl) in zip(axs.ravel(),F):
        ax.plot(z,f,color=BLUE,lw=1.8,label='$a = \\varphi(z)$'); ax.plot(z,df,color=RED,lw=1.5,ls='--',label="$\\varphi'(z)$")
        ax.axhline(0,color=GRID,lw=0.7); ax.axvline(0,color=GRID,lw=0.7); ax.set_ylim(-1.5,2.2) if nm!='logistic' else ax.set_ylim(-0.25,1.25)
        if nm=='identity': ax.set_ylim(-4.2,4.2)
        if nm=='relu': ax.set_ylim(-0.6,4.2)
        ax.set_title(f'{nm}:  {fl}',fontsize=9,color=NAVY); ax.grid(color=GRID,lw=0.5); ax.set_xlim(-4,4)
    axs[0,0].legend(fontsize=8.5,loc='upper left',handlelength=1.6,labelspacing=0.2); axs[1,0].set_xlabel('$z$'); axs[1,1].set_xlabel('$z$')
    axs[1,1].text(-3.8,2.9,"scikit-learn takes the\nderivative as 0 at $z=0$",fontsize=8.5,color=RED,ha='left',va='center')
    axs[0,1].text(1.0,0.45,'maximum 1/4\nat $z=0$',fontsize=8.5,color=RED,ha='left',va='center')
    fig.tight_layout(h_pad=0.5,w_pad=0.7); done(fig,'f331.png',4.8)

# ---- 31.10-1 gallery: hidden units x alpha
def moons_net(h,alpha,solver='lbfgs',seed=0,**kw): return make_pipeline(StandardScaler(),MLPClassifier(h,solver=solver,alpha=alpha,max_iter=1000,random_state=seed,**kw))
if want('f3101.png'):
    gx,gy,G=grid2((-2.0,3.1,-1.5,2.0),160)
    hs=[2,8,64]; als=[1e-4,1.0,3.0]; fig,axs=plt.subplots(3,3,figsize=(4.9,4.5),dpi=240,sharex=True,sharey=True); NUM['gallery']={}
    for i,h in enumerate(hs):
        for j,a in enumerate(als):
            ax=axs[i,j]; m=moons_net((h,),a).fit(Xm_tr,ym_tr); Z=m.predict(G).reshape(gx.shape); tr,te=m.score(Xm_tr,ym_tr),m.score(Xm_te,ym_te)
            ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],cmap=LBC,zorder=0); ax.contour(gx,gy,Z,levels=[0.5],colors=GREEN,linewidths=1.3,zorder=2)
            ax.scatter(Xm_tr[ym_tr==0,0],Xm_tr[ym_tr==0,1],s=5,color=BLUE,zorder=3); ax.scatter(Xm_tr[ym_tr==1,0],Xm_tr[ym_tr==1,1],s=5,color=OR,marker='s',zorder=3)
            ax.set_title(f'{h} units, $\\alpha$ = {a:g}\ntrain {tr:.2f}   test {te:.2f}',fontsize=8,color=NAVY,pad=2); ax.set_xticks([]); ax.set_yticks([])
            NUM['gallery'][f'{h}_{a:g}']=[round(tr,3),round(te,3)]
    fig.tight_layout(h_pad=0.4,w_pad=0.3); done(fig,'f3101.png',4.8)

# ---- 31.10-2 solvers and learning rates
if want('f3102.png'):
    Xs=StandardScaler().fit_transform(Xm_tr); fig,axs=plt.subplots(1,2,figsize=(4.9,2.3),dpi=240)
    ax=axs[0]; NUM['solvers']={}
    for nm,kw,c,ls in [('sgd, $\\eta$ = 0.01',dict(solver='sgd',learning_rate_init=0.01),PUR,'-'),('adam, $\\eta$ = 0.001',dict(solver='adam'),BLUE,'-')]:
        m=MLPClassifier((16,),max_iter=300,random_state=0,tol=0,n_iter_no_change=1000,**kw).fit(Xs,ym_tr); ax.plot(np.arange(1,len(m.loss_curve_)+1),m.loss_curve_,color=c,lw=1.6,ls=ls,label=nm); NUM['solvers'][nm]=[round(m.loss_curve_[k-1],4) for k in (10,50,300)]
    ks=list(range(1,61)); ls_=[]
    for k in ks:
        m=MLPClassifier((16,),solver='lbfgs',max_iter=k,random_state=0).fit(Xs,ym_tr); ls_.append(m.loss_)
    ax.plot(ks,ls_,color=GREEN,lw=1.6,label='lbfgs (per iteration)'); NUM['solvers']['lbfgs']=[round(ls_[k-1],4) for k in (10,50,60)]
    ax.set_yscale('log'); ax.set_xscale('log'); ax.set_xlabel('epochs  (L-BFGS: iterations)'); ax.set_ylabel('training loss $J$'); ax.grid(color=GRID,lw=0.5,which='both',alpha=0.6); ax.legend(fontsize=8.5,loc='upper right',handlelength=1.4,labelspacing=0.2,ncol=1); ax.set_ylim(0.1,3.2)
    ax.set_title('three solvers',fontsize=9,color=NAVY)
    ax=axs[1]; NUM['lr']={}
    for lr_,c in zip([1e-4,1e-3,1e-2,1e-1,1.0],plt.cm.Purples(np.linspace(0.3,0.95,5))):
        m=MLPClassifier((16,),max_iter=200,random_state=0,tol=0,n_iter_no_change=1000,learning_rate_init=lr_).fit(Xs,ym_tr); ax.plot(np.arange(1,201),m.loss_curve_,color=c,lw=1.5,label=f'{lr_:g}'); NUM['lr'][f'{lr_:g}']=[round(m.loss_curve_[19],4),round(m.loss_curve_[-1],4)]
    ax.set_yscale('log'); ax.set_xlabel('epochs'); ax.set_ylabel('training loss $J$'); ax.grid(color=GRID,lw=0.5,which='both',alpha=0.6); ax.set_title('Adam: $\\eta$ = 0.0001 ... 1',fontsize=9,color=NAVY)
    ax.legend(fontsize=8.3,title='$\\eta$',title_fontsize=8.3,loc='upper center',ncol=3,handlelength=1.2,labelspacing=0.1,borderaxespad=0.2,columnspacing=0.8); ax.set_ylim(0.12,6)
    fig.tight_layout(w_pad=0.8); done(fig,'f3102.png',4.8)

# ---- 31.10-3 validation curves
if want('f3103.png'):
    Xs=Xm_tr; fig,axs=plt.subplots(1,2,figsize=(4.9,2.2),dpi=240); NUM['valcurve']={}
    for ax,(nm,pname,prange,fix,xl) in zip(axs,[('alpha','mlpclassifier__alpha',np.logspace(-4,1,11),dict(mlpclassifier__hidden_layer_sizes=(64,)),'penalty $\\alpha$'),('hidden','mlpclassifier__hidden_layer_sizes',[(1,),(2,),(4,),(8,),(16,),(32,),(64,)],dict(mlpclassifier__alpha=0.1),'hidden units')]):
        TR=[];VA=[]
        for sd in range(4):
            pipe=make_pipeline(StandardScaler(),MLPClassifier(solver='lbfgs',max_iter=1000,random_state=sd)).set_params(**fix)
            tr,va=validation_curve(pipe,Xs,ym_tr,param_name=pname,param_range=prange,cv=5); TR.append(tr.mean(1)); VA.append(va.mean(1))
        xv=prange if nm=='alpha' else [p[0] for p in prange]
        mt,mv,sv=np.mean(TR,0),np.mean(VA,0),np.std(VA,0); ax.plot(xv,mt,color=BLUE,lw=1.6,marker='o',ms=3,label='training folds'); ax.plot(xv,mv,color=GREY,lw=1.6,ls='--',marker='s',ms=3,label='validation folds'); ax.fill_between(xv,mv-sv,mv+sv,color=GREY,alpha=0.15,lw=0)
        kb=int(np.argmax(mv)); ax.axvline(xv[kb],color=PUR,lw=1.1); ax.set_xscale('log'); ax.set_xlabel(xl); ax.set_ylabel('accuracy'); ax.grid(color=GRID,lw=0.5); NUM['valcurve'][nm]={'best':float(xv[kb]),'val':[round(float(v),3) for v in mv],'train':[round(float(v),3) for v in mt]}
    axs[0].legend(fontsize=8,loc='lower left',handlelength=1.6,labelspacing=0.2); axs[0].set_title('64 hidden units',fontsize=9,color=NAVY); axs[1].set_title('$\\alpha$ = 0.1',fontsize=9,color=NAVY); axs[1].set_ylabel('')
    axs[1].set_xticks([1,2,4,8,16,32,64]); axs[1].set_xticklabels([1,2,4,8,16,32,64]); axs[1].minorticks_off()
    fig.tight_layout(w_pad=0.8); done(fig,'f3103.png',4.8)

# ---- 31.6-1 forward (blue) and backward (red) pass of the hand example
if want('f361.png'):
    W1=np.array([[0.5,1.0],[-1.0,0.5]]); b1=np.array([0.5,-0.5]); W2=np.array([[1.0],[-1.0]]); b2=0.0; x=np.array([2.0,-1.0])
    z1=x@W1+b1; h=np.maximum(z1,0); z2=float(h@W2[:,0]+b2); p=1/(1+np.exp(-z2)); d2=p-1; d1=(W2[:,0]*d2)*(z1>0)
    NUM['hand']={'z1':z1.tolist(),'z2':z2,'p':round(float(p),4),'d2':round(float(d2),4),'d1':[round(float(v),4) for v in d1]}
    fig,ax=plt.subplots(figsize=(4.9,3.3),dpi=240); ax.axis('off'); ax.set_xlim(-0.3,10.2); ax.set_ylim(-3.0,4.6); ax.set_aspect('equal')
    pos={'x1':(1.0,3.2),'x2':(1.0,0.4),'h1':(5.0,3.2),'h2':(5.0,0.4),'o':(9.0,1.8)}
    def edge(a,b_,lab,t=0.38,dy=0.0):
        (x0,y0),(x1_,y1_)=pos[a],pos[b_]; ax.plot([x0+0.45,x1_-0.45],[y0,y1_],color=GRID,lw=1.2,zorder=1,solid_capstyle='round')
        ax.text(x0+(x1_-x0)*t+0.2,y0+(y1_-y0)*t+dy,lab,fontsize=8.5,color=BLUE,ha='center',va='center',bbox=dict(fc='white',ec='none',pad=0.8),zorder=3)
    edge('x1','h1','0.5',0.42,0.1); edge('x1','h2','1.0',0.22,0.1); edge('x2','h1','−1.0',0.22,-0.1); edge('x2','h2','0.5',0.42,-0.1); edge('h1','o','1.0',0.45); edge('h2','o','−1.0',0.45)
    for k,(nm,(xx,yy)) in enumerate(pos.items()):
        c=BLUE if nm[0]=='x' else (GREEN if nm[0]=='h' else OR); ax.add_patch(Circle((xx,yy),0.48,fc='white',ec=c,lw=1.8,zorder=4))
    ax.text(*pos['x1'],'2',ha='center',va='center',fontsize=10,color=NAVY,zorder=5); ax.text(*pos['x2'],'−1',ha='center',va='center',fontsize=10,color=NAVY,zorder=5)
    def block(nm,b,z,d):
        xx,yy=pos[nm]
        ax.text(xx,yy-0.85,f'$b$ = {b:g}'.replace('-','−'),ha='center',va='center',fontsize=8.5,color=BLUE)
        ax.text(xx,yy-1.30,f'$z$ = {z:g}',ha='center',va='center',fontsize=8.5,color=BLUE)
        ax.text(xx,yy-1.75,f'$\\delta$ = {d:+.4f}'.replace('-','−'),ha='center',va='center',fontsize=8.5,color=RED)
    for j,nm in enumerate(['h1','h2']):
        ax.text(*pos[nm],f'{h[j]:g}',ha='center',va='center',fontsize=10,color=NAVY,zorder=5); block(nm,b1[j],z1[j],d1[j])
    ax.text(*pos['o'],f'{p:.2f}',ha='center',va='center',fontsize=9,color=NAVY,zorder=5); block('o',0.0,z2,d2)
    ax.annotate('',xy=(1.6,-2.75),xytext=(8.4,-2.75),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.2)); ax.text(5.0,-2.4,'backward pass: $\\delta$ is passed from the loss to the weights',fontsize=8.5,color=RED,ha='center',va='center')
    ax.annotate('',xy=(8.4,4.15),xytext=(1.6,4.15),arrowprops=dict(arrowstyle='-|>',color=BLUE,lw=1.2)); ax.text(5.0,4.5,'forward pass:  $\\hat{y}$ = 0.8176,   $y$ = 1',fontsize=8.5,color=BLUE,ha='center',va='center')
    fig.subplots_adjust(0,0,1,1); done(fig,'f361.png',4.9)

# ---- 31.7-1 flow chart of fit
if want('f371.png'):
    fig,ax=plt.subplots(figsize=(4.9,3.6),dpi=240); ax.axis('off'); ax.set_xlim(-0.5,10); ax.set_ylim(0,10.4)
    def bx(x,y,w,h,t,fc,ec,fs=8.5): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=fs)
    bx(2.2,9.3,5.6,1.0,'validate X, y; encode the labels;\nchoose the output activation',LB,BLUE)
    bx(2.2,7.9,5.6,0.9,'initialise W and b, uniform in ± a   (Eq. 31.5-1)',LB,BLUE)
    bx(0.1,4.9,4.5,2.0,'adam or sgd, each epoch:\nshuffle; for each mini-batch:\nforward, loss, backward,\nupdate (Eq. 31.5-2 or 31.5-4)','#E3F4EA',GREEN)
    bx(5.4,5.4,4.5,1.5,'lbfgs: L-BFGS-B on the\nfull-batch loss, at most\nmax_iter steps','#F1EAF7',PUR)
    bx(0.1,2.6,4.5,1.5,'stop?  no gain of tol for\nn_iter_no_change + 1 epochs,\nor max_iter epochs done','#FDF0E0','#E67E22')
    bx(2.2,0.5,5.6,1.0,'store coefs_, intercepts_, loss_,\nloss_curve_, n_iter_, t_',LB,BLUE)
    arr(ax,(5,9.3),(5,8.85),c=NAVY); arr(ax,(3.4,7.9),(2.8,6.95),c=NAVY); arr(ax,(6.6,7.9),(7.6,6.95),c=NAVY)
    arr(ax,(2.35,4.9),(2.35,4.15),c=NAVY); arr(ax,(3.4,2.6),(4.0,1.55),c=NAVY); arr(ax,(6.4,5.4),(6.4,1.55),c=NAVY)
    ax.plot([0.1,-0.25,-0.25],[3.35,3.35,5.9],color='#E67E22',lw=1.2); arr(ax,(-0.25,5.9),(0.1,5.9),c='#E67E22')
    ax.text(-0.15,4.55,'no',fontsize=8.5,color='#E67E22',ha='left',va='center',rotation=90); ax.text(3.5,2.05,'yes',fontsize=8.5,color='#E67E22',ha='right',va='center')
    ax.text(6.7,3.2,'with early_stopping,\n10 % of the rows are\nheld out and the stop\nrule uses their score',fontsize=8,color=GREY,ha='left',va='center')
    fig.subplots_adjust(0,0,1,1); done(fig,'f371.png',4.9)

# ---- 31.11-1 RBM components
if want('f3111.png'):
    Xd,yd=load_digits(return_X_y=True); Xd=Xd/16.0
    Xtr_,Xte_,ytr_,yte_=train_test_split(Xd,yd,test_size=0.25,random_state=0,stratify=yd)
    rbm=BernoulliRBM(n_components=100,learning_rate=0.1,n_iter=30,random_state=0).fit(Xtr_)
    Wc=rbm.components_[np.argsort(-np.linalg.norm(rbm.components_,axis=1))]; v=np.abs(Wc).max()
    fig,axs=plt.subplots(3,10,figsize=(4.9,1.75),dpi=240)
    for k in range(10):
        axs[0,k].imshow(Xte_[k].reshape(8,8),cmap='Greys',vmin=0,vmax=1); axs[0,k].set_title(str(yte_[k]),fontsize=8,pad=1.5,color=GREY)
        for r_ in (1,2): w_=Wc[(r_-1)*10+k]; vm=np.abs(w_).max(); axs[r_,k].imshow(w_.reshape(8,8),cmap='RdBu_r',vmin=-vm,vmax=vm)
    for ax in axs.ravel(): ax.set_xticks([]); ax.set_yticks([]); [sp.set_visible(False) for sp in ax.spines.values()]
    for r_,t in enumerate(['test digits','components\n1 to 10','components\n11 to 20']): axs[r_,0].set_ylabel(t,fontsize=8,rotation=0,ha='right',va='center',labelpad=3)
    NUM['rbm']={'max_abs_weight':round(float(v),2),'h_mean_active':round(float(rbm.transform(Xte_).mean()),3)}
    fig.subplots_adjust(left=0.13,right=0.995,top=0.93,bottom=0.01,wspace=0.06,hspace=0.08); done(fig,'f3111.png',4.9)

# ---- 31.12-1 depth: tent maps, and gradients per layer
if want('f3121.png'):
    relu=lambda z:np.maximum(z,0); tent=lambda x:2*relu(x)-4*relu(x-0.5); xg=np.linspace(0,1,2049)
    fig=plt.figure(figsize=(4.9,3.5),dpi=240); gs=fig.add_gridspec(2,3,height_ratios=[1,1.15],hspace=0.55,wspace=0.35)
    f=xg.copy()
    for L in range(1,4):
        f=tent(f); ax=fig.add_subplot(gs[0,L-1]); ax.plot(xg,f,color=GREEN,lw=1.6)
        ax.set_title(f'{L} hidden layer'+('s' if L>1 else '')+f': {2**L} pieces',fontsize=8.5,color=NAVY); ax.set_xticks([0,0.5,1]); ax.set_yticks([0,1]); ax.set_xlabel('$x$',labelpad=1); ax.grid(color=GRID,lw=0.5)
        if L==1: ax.set_ylabel('$g^{L}(x)$')
    sys.path.insert(0,'code'); from depth import layer_gradients
    ax=fig.add_subplot(gs[1,:]); NUM['grad']={}
    for act,c,ls,lab in [('logistic',RED,'-','logistic'),('tanh',PUR,'--','tanh'),('relu',GREEN,'-','relu')]:
        g=layer_gradients(act); ax.semilogy(np.arange(1,12),g,color=c,lw=1.7,ls=ls,marker='o',ms=3,label=lab); NUM['grad'][act]=[float(f'{v:.2e}') for v in g]
    ax.set_xlabel('layer  (1 = next to the input, 11 = output layer)'); ax.set_ylabel('RMS gradient of $W^{(l)}$'); ax.grid(color=GRID,lw=0.5,which='both',alpha=0.6); ax.legend(fontsize=8,loc='lower right',ncol=3,handlelength=1.6,columnspacing=1.0)
    ax.set_xticks(range(1,12)); ax.set_title('10 hidden layers of 30 units, at initialisation',fontsize=8.5,color=NAVY)
    done(fig,'f3121.png',4.9)

# ---- 31.13-1 digits: loss curve and the mistakes
if want('f3131.png'):
    Xd,yd=load_digits(return_X_y=True); Xtr_,Xte_,ytr_,yte_=train_test_split(Xd,yd,test_size=0.25,random_state=0,stratify=yd)
    net=make_pipeline(StandardScaler(),MLPClassifier(random_state=0)).fit(Xtr_,ytr_); mlp=net[-1]; pred=net.predict(Xte_); bad=np.where(pred!=yte_)[0]
    NUM['digits']={'errors':int(len(bad)),'n_iter':int(mlp.n_iter_),'final_loss':round(float(mlp.loss_),5),'confused':[[int(yte_[i]),int(pred[i])] for i in bad]}
    fig=plt.figure(figsize=(4.9,2.3),dpi=240); gs=fig.add_gridspec(2,7,width_ratios=[1,1,1,1,1,0.85,3.0],hspace=0.45,wspace=0.12)
    for k,i in enumerate(bad[:10]):
        ax=fig.add_subplot(gs[k//5,k%5]); ax.imshow(Xte_[i].reshape(8,8),cmap='Greys'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'{yte_[i]} → {pred[i]}',fontsize=8.5,pad=2,color=RED)
        for sp in ax.spines.values(): sp.set_visible(False)
    ax=fig.add_subplot(gs[:,6]); lc=np.array(mlp.loss_curve_); ax.semilogy(np.arange(1,len(lc)+1),lc,color=BLUE,lw=1.7); ax.set_xlabel('epoch'); ax.set_ylabel('training loss $J$'); ax.grid(color=GRID,lw=0.5,which='both',alpha=0.6)
    ax.axvline(mlp.n_iter_,color=PUR,lw=1.0,ls='--'); ax.text(mlp.n_iter_-3,lc[3],f'stops after\n{mlp.n_iter_} epochs',fontsize=8,color=PUR,ha='right',va='center')
    fig.text(0.02,0.97,f'{len(bad)} of 450 test digits wrong (true → predicted)',fontsize=8.5,color=RED,ha='left',va='top')
    fig.subplots_adjust(left=0.01,right=0.995,top=0.86,bottom=0.17); done(fig,'f3131.png',4.9)

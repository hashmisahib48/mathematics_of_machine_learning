# Problem figures of Chapter 31. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from figstyle import *
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from sklearn.datasets import make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore',ConvergenceWarning)
A={}
X8=np.array([[-2,-1],[-1,-2],[1,2],[2,1],[-2,1],[-1,2],[1,-2],[2,-1]],float); y8=np.array([0,0,0,0,1,1,1,1])
W1=np.array([[0.5,1.0],[-1.0,0.5]]); b1=np.array([0.5,-0.5]); W2=np.array([1.0,-1.0]); b2=0.0
sig=lambda z:1/(1+np.exp(-z))
def fwd(x):
    z1=x@W1+b1; h=np.maximum(z1,0); z2=h@W2+b2; return z1,h,z2,sig(z2)
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def xy(ax,pts=None,hl=None):
    ax.scatter(X8[y8==0,0],X8[y8==0,1],s=22,color=BLUE,edgecolor='white',lw=0.6,zorder=4); ax.scatter(X8[y8==1,0],X8[y8==1,1],s=22,color=OR,marker='s',edgecolor='white',lw=0.6,zorder=4)
    ax.axhline(0,color=GRID,lw=0.7); ax.axvline(0,color=GRID,lw=0.7); ax.set_xlim(-2.9,2.9); ax.set_ylim(-2.9,2.9); ax.set_xticks([-2,0,2]); ax.set_yticks([-2,0,2]); ax.set_xlabel('$x_1$',labelpad=0); ax.set_ylabel('$x_2$',labelpad=0); ax.set_aspect('equal')
def fmt(v,k=4): return round(float(v),k)

# ---- 31A.1 forward pass of another sample
x=np.array([2.0,1.0]); z1,h,z2,p=fwd(x); A['31A.1']={'x':x.tolist(),'z1':z1.tolist(),'h':h.tolist(),'z2':fmt(z2),'p':fmt(p),'loss':fmt(-np.log(1-p))}
fig,ax=small(2.9,1.5); xy(ax); ax.scatter([2],[1],s=70,facecolor='none',edgecolor=RED,lw=1.5,zorder=6); ax.annotate('(2, 1), $y$ = 0',xy=(2,1),xytext=(0.15,0.3),fontsize=9.4,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.8)); save(fig,'p_31A1.png')
# ---- 31A.2 dead unit
x=np.array([1.0,2.0]); y=0; z1,h,z2,p=fwd(x); d2=p-y; d1=(W2*d2)*(z1>0)
A['31A.2']={'z1':z1.tolist(),'h':h.tolist(),'p':fmt(p),'d2':fmt(d2),'d1':[fmt(v) for v in d1],'gW2':[fmt(v) for v in h*d2],'gb2':fmt(d2),'gW1':[[fmt(v) for v in r] for r in np.outer(x,d1)],'gb1':[fmt(v) for v in d1]}
fig,ax=small(2.3,1.2); zz=np.linspace(-2.3,2.3,100); ax.plot(zz,np.maximum(zz,0),color=BLUE,lw=1.6)
for zi,c,nm in [(z1[0],RED,'unit 1'),(z1[1],GREEN,'unit 2')]: ax.plot([zi],[max(zi,0)],'o',color=c,ms=6,zorder=5); ax.text(zi-0.15 if zi>0 else zi+0.1,max(zi,0)+(0.15 if zi>0 else 0.35),f'{nm}: $z$ = {zi:g}',fontsize=9.4,color=c,ha='right' if zi>0 else 'center',va='bottom')
ax.axhline(0,color=GRID,lw=0.7); ax.set_xlabel('$z$',labelpad=0); ax.set_ylabel('$a$',labelpad=0); ax.set_xlim(-2.3,2.3); ax.set_ylim(-0.3,2.4); ax.set_xticks([-2,0,2]); save(fig,'p_31A2.png')
# ---- 31A.3 shapes and softmax
sizes=[64,100,50,10]; shapes=list(zip(sizes[:-1],sizes[1:])); npar=sum(a*b+b for a,b in shapes)
zs=np.array([2.0,0.0,-1.0]); P=np.exp(zs)/np.exp(zs).sum(); A['31A.3']={'shapes':shapes,'n_params':int(npar),'per_layer':[a*b+b for a,b in shapes],'P':[fmt(v) for v in P],'loss_class2':fmt(-np.log(P[1]))}
fig,ax=new(3.0,1.75)
xs=[0.2,1.4,2.6,3.8]
for k,(xx,n) in enumerate(zip(xs,sizes)): box(ax,xx,0.8,0.8,0.7,str(n),fc=LB if k<3 else '#FDF0E0',ec=BLUE if k<3 else OR,fs=10)
for k in range(3):
    arr(ax,(xs[k]+0.8,1.15),(xs[k+1],1.15),c=NAVY); ax.text(xs[k]+1.1+0.1,1.62,f'({sizes[k]}, {sizes[k+1]})',fontsize=9.4,color=GREEN,ha='center',va='bottom') if False else None
for k in range(3): ax.text((xs[k]+0.8+xs[k+1])/2,1.62,f'{sizes[k]}×{sizes[k+1]}',fontsize=9.4,color=GREEN,ha='center',va='bottom')
ax.text(2.4,0.25,'coefs_[0], coefs_[1], coefs_[2]',fontsize=9.4,color=GREY,ha='center'); ax.set_xlim(0,4.8); ax.set_ylim(0.1,2.0); save(fig,'p_31A3.png')
# ---- 31A.4 momentum
g=[1.0,1.0,-2.0]; eta,mu=0.1,0.5; v=0.0; w1=0.0; w2=0.0; hv=[];nv=[]
for gt in g:
    v=mu*v-eta*gt; w1+=v; hv.append(w1); w2+=mu*v-eta*gt; nv.append(w2)
plain=list(np.cumsum([-eta*gt for gt in g]))
ref=[]; w=[np.zeros(1)]; o=SGDOptimizer(w,eta,'constant',mu,False)
for gt in g: o.update_params(w,[np.array([gt])]); ref.append(w[0][0])
ref2=[]; w=[np.zeros(1)]; o=SGDOptimizer(w,eta,'constant',mu,True)
for gt in g: o.update_params(w,[np.array([gt])]); ref2.append(w[0][0])
assert np.allclose(ref,hv) and np.allclose(ref2,nv)
A['31A.4']={'plain':[fmt(v) for v in plain],'momentum':[fmt(v) for v in hv],'nesterov':[fmt(v) for v in nv]}
fig,ax=small(2.3,1.3); t=np.arange(0,4)
ax.plot(t,[0]+plain,'-o',color=GREY,ms=3.5,lw=1.4,label='plain'); ax.plot(t,[0]+hv,'-s',color=PUR,ms=3.5,lw=1.4,label='momentum'); ax.plot(t,[0]+nv,'--^',color=GREEN,ms=3.5,lw=1.4,label='Nesterov')
ax.axhline(0,color=GRID,lw=0.7); ax.set_xlabel('step',labelpad=0); ax.set_ylabel('$w$',labelpad=0); ax.set_xticks(t); ax.legend(fontsize=9.4,handlelength=1.6,labelspacing=0.2,ncol=1,loc='lower left',handletextpad=0.3,borderaxespad=0.1); ax.set_ylim(-0.5,0.2); save(fig,'p_31A4.png')

# ---- 31B.1 gradient check with the penalty
exec(open('code/mlp_scratch.py').read())
Xi,yi=load_iris(return_X_y=True); Xi=StandardScaler().fit_transform(Xi); Yi=(yi[:,None]==np.unique(yi))*1.0
def gradcheck(alpha,drop=False):
    net=MyMLP(alpha=alpha,n_iter=5).fit(Xi,yi); par=(net.W1,net.b1,net.W2,net.b2); G=net.loss_grad(Xi,Yi)[1]; worst=0; an=[];nu=[]
    if drop: G=[(Xi.T@((G[0]*0)) )]  # placeholder, replaced below
    return net,par,G
res={}; pts={}
for alpha in (0.0,3.0):
    net=MyMLP(alpha=alpha,n_iter=5).fit(Xi,yi); par=(net.W1,net.b1,net.W2,net.b2); G=net.loss_grad(Xi,Yi)[1]; an=[];nu=[]
    for q,gq in zip(par,G):
        for k in np.ndindex(q.shape):
            old=q[k]; q[k]=old+1e-6; Jp=net.loss_grad(Xi,Yi)[0]; q[k]=old-1e-6; Jm=net.loss_grad(Xi,Yi)[0]; q[k]=old; nu.append((Jp-Jm)/2e-6); an.append(gq[k])
    an=np.array(an); nu=np.array(nu); res[alpha]=float(abs(an-nu).max()); pts[alpha]=(an,nu)
    if alpha==3.0:   # the same check with the penalty term left out of the gradient
        pen=np.concatenate([(3.0*net.W1/len(Xi)).ravel(),np.zeros(net.b1.size),(3.0*net.W2/len(Xi)).ravel(),np.zeros(net.b2.size)]); an2=an-pen; res['3.0_no_penalty_term']=float(abs(an2-nu).max()); pts['np']=(an2,nu)
A['31B.1']={'largest_error_alpha0':float(f'{res[0.0]:.1e}'),'largest_error_alpha3':float(f'{res[3.0]:.1e}'),'error_without_penalty_term':float(f'{res["3.0_no_penalty_term"]:.1e}')}
fig,ax=small(2.4,1.1); an,nu=pts[3.0]; an2,_=pts['np']; idx=np.arange(len(an))
ax.semilogy(idx,np.maximum(abs(an2-nu),1e-13),'.',color=RED,ms=4,label='term left out'); ax.semilogy(idx,np.maximum(abs(an-nu),1e-13),'.',color=GREEN,ms=4,label='complete')
ax.set_xlabel('parameter number',labelpad=0); ax.set_ylabel('absolute error',labelpad=0); ax.legend(fontsize=9.4,loc='center right',handletextpad=0.1,labelspacing=0.2,borderaxespad=0.1); ax.set_ylim(1e-13,1); ax.tick_params(labelsize=9.4); save(fig,'p_31B1.png')

# ---- 31B.2 tanh hidden layer
class TanhMLP(MyMLP):
    def forward(self,X):
        Z1=X@self.W1+self.b1; H=np.tanh(Z1); Z2=H@self.W2+self.b2; E=np.exp(Z2-Z2.max(axis=1,keepdims=True)); return Z1,H,E/E.sum(axis=1,keepdims=True)
    def loss_grad(self,X,Y):
        n,a=len(X),self.alpha; Z1,H,P=self.forward(X); pen=(self.W1**2).sum()+(self.W2**2).sum(); J=-np.log((P*Y).sum(axis=1)).mean()+a*pen/(2*n)
        D2=P-Y; D1=(D2@self.W2.T)*(1-H**2)
        return J,[(X.T@D1+a*self.W1)/n,D1.mean(axis=0),(H.T@D2+a*self.W2)/n,D2.mean(axis=0)]
mine=TanhMLP(alpha=0.1,eta=0.5,n_iter=100).fit(Xi,yi)
ref=MLPClassifier((8,),activation='tanh',solver='sgd',momentum=0,alpha=0.1,learning_rate_init=0.5,batch_size=len(Xi),max_iter=100,tol=0,random_state=0).fit(Xi,yi)
A['31B.2']={'loss_mine':fmt(mine.loss_curve_[-1],6),'loss_sklearn':fmt(ref.loss_curve_[-1],6),'max_diff_W1':float(f'{abs(mine.W1-ref.coefs_[0]).max():.1e}'),'same_predictions':bool((mine.predict(Xi)==ref.predict(Xi)).all()),'relu_loss_100':fmt(MyMLP(alpha=0.1,eta=0.5,n_iter=100).fit(Xi,yi).loss_curve_[-1],6)}
fig,ax=small(2.3,1.3); rl=MyMLP(alpha=0.1,eta=0.5,n_iter=100).fit(Xi,yi).loss_curve_
ax.plot(rl,color=GREY,lw=1.4,ls='--',label='ReLU'); ax.plot(mine.loss_curve_,color=PUR,lw=1.8,label='tanh, yours'); ax.plot(ref.loss_curve_,color=GREEN,lw=1.0,ls=':',label='scikit-learn')
ax.set_yscale('log'); ax.set_xlabel('iteration',labelpad=0); ax.set_ylabel('loss $J$',labelpad=0); ax.legend(fontsize=9.4,loc='upper right',handlelength=1.6,labelspacing=0.2); ax.tick_params(labelsize=9.4); save(fig,'p_31B2.png')

# ---- 31B.3 early stopping
Xm,ym=make_moons(100,noise=0.3,random_state=0)
m=MLPClassifier((50,),early_stopping=True,n_iter_no_change=5,validation_fraction=0.2,random_state=0,max_iter=500,alpha=0).fit(Xm,ym)
vs=np.array(m.validation_scores_); best=int(vs.argmax())+1
A['31B.3']={'n_iter':int(m.n_iter_),'best_epoch':best,'best_validation_score':fmt(m.best_validation_score_,3),'n_validation_rows':20,'t_':int(m.t_),'best_loss_':None if m.best_loss_ is None else float(m.best_loss_)}
fig,ax=small(2.3,2.05); ax.plot(np.arange(1,len(vs)+1),vs,'-o',color=GREY,ms=3,lw=1.4); ax.plot([best],[vs.max()],'o',color=PUR,ms=7,zorder=5); ax.axvline(m.n_iter_,color=RED,lw=1,ls='--')
ax.text(best-0.9,vs.max()+0.02,f'best: {best}',fontsize=9.4,color=PUR,ha='right'); ax.text(m.n_iter_-0.3,vs.min()+0.01,f'stop: {m.n_iter_}',fontsize=9.4,color=RED,ha='right',va='bottom')
ax.set_xlabel('epoch',labelpad=0); ax.set_ylabel('validation accuracy',labelpad=0); ax.tick_params(labelsize=9.4); ax.set_ylim(vs.min()-0.03,vs.max()+0.06); save(fig,'p_31B3.png')

# ---- 31C.1 identical start
Xs=StandardScaler().fit_transform(Xm)
def run(sym):
    n=MLPClassifier((2,),solver='sgd',momentum=0,learning_rate_init=0.5,batch_size=len(Xs),max_iter=1,random_state=0,alpha=0).fit(Xs,ym)
    if sym:
        n.coefs_[0][:]=np.array([[0.4,0.4],[-0.3,-0.3]]); n.intercepts_[0][:]=0.1; n.coefs_[1][:]=0.5; n.intercepts_[1][:]=0.0
    d=[]
    for _ in range(60):
        n.partial_fit(Xs,ym,classes=[0,1]); d.append(np.linalg.norm(n.coefs_[0][:,0]-n.coefs_[0][:,1]))
    return d
ds,dr=run(True),run(False); A['31C.1']={'diff_after_60_symmetric':float(f'{ds[-1]:.1e}'),'diff_after_60_random':fmt(dr[-1],3)}
fig,ax=small(2.3,1.3); ax.semilogy(range(1,61),np.maximum(ds,1e-16),color=RED,lw=1.6,label='equal start'); ax.semilogy(range(1,61),dr,color=GREEN,lw=1.6,label='random start')
ax.set_xlabel('update',labelpad=0); ax.set_ylabel('distance',labelpad=0); ax.legend(fontsize=9.4,loc='center right',handlelength=1.6,labelspacing=0.2); ax.tick_params(labelsize=9.4); ax.set_ylim(1e-18,10); ax.set_yticks([1e-16,1e-8,1]); ax.text(2,3e-16,'exactly 0',fontsize=9.4,color=RED,ha='left',va='bottom'); save(fig,'p_31C1.png')

# ---- 31C.2 beyond the data
rng=np.random.default_rng(1); xt=np.sort(rng.uniform(-3,3,40)); yt=np.sin(xt)+rng.normal(0,0.1,40); xg=np.linspace(-6,6,300)
fig,ax=small(2.5,1.4); ax.axvspan(-3,3,color='#E3EEFB',zorder=0,lw=0); ax.scatter(xt,yt,s=7,color=BLUE,zorder=4)
A['31C.2']={}
for act,c,ls in [('relu',GREEN,'-'),('tanh',PUR,'--')]:
    best=None
    for s in range(10):
        mm=MLPRegressor(hidden_layer_sizes=(10,),activation=act,solver='lbfgs',alpha=1e-3,max_iter=3000,random_state=s).fit(xt[:,None],yt)
        if best is None or mm.loss_<best.loss_: best=mm
    ax.plot(xg,best.predict(xg[:,None]),color=c,lw=1.5,ls=ls,label=act); A['31C.2'][act]=[fmt(best.predict([[v]])[0],2) for v in (-6,0,6)]
ax.set_ylim(-3.5,3.5); ax.set_xlabel('$x$',labelpad=0); ax.set_ylabel('prediction',labelpad=0); ax.legend(fontsize=9.4,loc='upper right',handlelength=1.5,labelspacing=0.2); ax.tick_params(labelsize=9.4); save(fig,'p_31C2.png')

# ---- 31C.3 which epoch?
Xd,yd=make_moons(400,noise=0.5,random_state=3); Xd=StandardScaler().fit_transform(Xd); Xtr,ytr,Xva,yva=Xd[:60],yd[:60],Xd[60:],yd[60:]
net=MLPClassifier((100,),solver='adam',alpha=0,random_state=0,learning_rate_init=0.01,batch_size=60); tl=[];vl=[]
from sklearn.metrics import log_loss
for ep in range(300):
    net.partial_fit(Xtr,ytr,classes=[0,1]); tl.append(log_loss(ytr,net.predict_proba(Xtr))); vl.append(log_loss(yva,net.predict_proba(Xva)))
bi=int(np.argmin(vl))+1; A['31C.3']={'n_train':60,'n_val':340,'best_epoch':bi,'val_loss_best':fmt(min(vl),3),'val_loss_end':fmt(vl[-1],3),'train_loss_end':fmt(tl[-1],3)}
fig,ax=small(2.3,1.3); ax.plot(range(1,301),tl,color=BLUE,lw=1.5,label='training'); ax.plot(range(1,301),vl,color=GREY,lw=1.5,ls='--',label='validation'); ax.axvline(bi,color=PUR,lw=1); ax.text(bi+8,1.0,f'min at {bi}',fontsize=9.4,color=PUR,ha='left',va='top')
ax.set_xlabel('epoch',labelpad=0); ax.set_ylabel('log loss',labelpad=0); ax.legend(fontsize=9.4,loc='upper left',bbox_to_anchor=(0.12,1.0),handlelength=1.6,labelspacing=0.2); ax.tick_params(labelsize=9.4); ax.set_ylim(0,2.1); save(fig,'p_31C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

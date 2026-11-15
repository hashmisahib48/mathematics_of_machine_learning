# Problem figures of Chapter 17. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
import warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.datasets import load_breast_cancer, load_iris, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sgd_scratch import SLOPE, rate
A={}
TX=np.array([[4,3],[1,0],[3,4],[0,1]],float); TY=np.array([1,-1,1,-1])
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def small(w=2.4,h=1.7): return plt.subplots(figsize=(w,h),dpi=240)
def cls(ax,X,y,s=30):
    ax.scatter(X[y<0,0],X[y<0,1],s=s,color=BLUE,edgecolor='white',lw=0.8,zorder=5,marker='o'); ax.scatter(X[y>0,0],X[y>0,1],s=s,color=OR,edgecolor='white',lw=0.8,zorder=5,marker='s')
def boundary(ax,w,b,xl,yl,**k):
    x0,x1=xl; y0,y1=yl; P=[]
    if abs(w[1])>1e-12:
        for xx in (x0,x1):
            yy=-(w[0]*xx+b)/w[1]
            if y0-1e-9<=yy<=y1+1e-9: P.append((xx,yy))
    if abs(w[0])>1e-12:
        for yy in (y0,y1):
            xx=-(w[1]*yy+b)/w[0]
            if x0-1e-9<=xx<=x1+1e-9: P.append((xx,yy))
    P=sorted(set(P))
    if len(P)>=2: ax.plot([P[0][0],P[-1][0]],[P[0][1],P[-1][1]],**k)
def prun(order,eta=1.0,epochs=50):
    w=np.zeros(2); b=0.0; log=[]
    for e in range(epochs):
        m=0
        for i in order:
            z=TY[i]*(TX[i]@w+b)
            if z<=0: w=w+eta*TY[i]*TX[i]; b=b+eta*TY[i]; m+=1; log.append((e+1,i+1))
        if m==0: break
    return w,b,log

# ---- 17A.1 another order
order=[2,0,3,1]; w,b,log=prun(order); A['17A.1']={'order_1based':[o+1 for o in order],'updates':len(log),'updates_list':log,'w':w.tolist(),'b':b}
fig,ax=small(2.4,1.8); cls(ax,TX,TY,34)
for k,o in enumerate(order): ax.text(TX[o,0]+0.18,TX[o,1]+0.12,str(k+1),fontsize=9.5,color=NAVY,weight='bold')
boundary(ax,w,b,(-0.6,5.4),(-0.6,5.4),color=GREEN,lw=1.6); ax.text(-0.45,5.25,'numbers: order\nof visiting',fontsize=9,color=GREY,va='top')
ax.set_xlim(-0.6,5.4); ax.set_ylim(-0.6,5.4); grid(ax); ax.set_xlabel('$x_1$',fontsize=9.5); ax.set_ylabel('$x_2$',fontsize=9.5); ax.tick_params(labelsize=9); fig.tight_layout(); save(fig,'p_17A1.png')

# ---- 17A.2 two candidate boundaries, losses and objective
def losses(w,b):
    z=TY*(TX@w+b); return z,{'hinge':np.maximum(0,1-z).mean(),'squared_hinge':(np.maximum(0,1-z)**2).mean(),'log_loss':np.log1p(np.exp(-z)).mean(),'perceptron':np.maximum(0,-z).mean()}
za,La=losses(np.array([2.,1.]),-3.0); zb,Lb=losses(np.array([0.3,0.3]),-1.2); al=0.1
A['17A.2']={'a':{'z':za.tolist(),**{k:round(float(v),4) for k,v in La.items()},'J_hinge':round(float(La['hinge']+al/2*5),4)},'b':{'z':[round(float(v),4) for v in zb],**{k:round(float(v),4) for k,v in Lb.items()},'J_hinge':round(float(Lb['hinge']+al/2*0.18),4)},'alpha':al}
fig,ax=small(2.4,1.8); cls(ax,TX,TY,30); xl=(-0.6,5.4)
boundary(ax,np.array([2.,1.]),-3.0,xl,xl,color=PUR,lw=1.5,ls='--'); boundary(ax,np.array([0.3,0.3]),-1.2,xl,xl,color=GREEN,lw=1.5)
ax.text(5.3,6.0,'(a) dashed',color=PUR,fontsize=9,ha='right'); ax.text(5.3,5.2,'(b) solid',color=GREEN,fontsize=9,ha='right')
ax.set_xlim(*xl); ax.set_ylim(xl[0],6.6); grid(ax); ax.set_xlabel('$x_1$',fontsize=9.5); ax.set_ylabel('$x_2$',fontsize=9.5); ax.tick_params(labelsize=9); fig.tight_layout(); save(fig,'p_17A2.png')

# ---- 17A.3 two log-loss steps, checked with SGDClassifier
w=np.zeros(2); b=0.0; eta=0.5; steps=[]
for i in (0,1):
    z=TY[i]*(TX[i]@w+b); Lp=-1/(1+np.exp(z)); w=w-eta*Lp*TY[i]*TX[i]; b=b-eta*Lp*TY[i]; steps.append({'z':round(float(z),4),'slope':round(float(Lp),4),'w':[round(float(v),4) for v in w],'b':round(float(b),4)})
m=SGDClassifier(loss='log_loss',penalty=None,learning_rate='constant',eta0=eta,shuffle=False,random_state=0); m.partial_fit(TX[:2],(TY[:2]+1)//2,classes=[0,1])
A['17A.3']={'steps':steps,'sklearn_w':m.coef_[0].round(4).tolist(),'sklearn_b':m.intercept_.round(4).tolist()}
fig,ax=small(2.4,1.7); zz=np.linspace(-3,3,300); ax.plot(zz,-1/(1+np.exp(zz)),color=GREEN,lw=1.6); ax.axhline(0,color=GRID,lw=0.8)
for st,c_ in zip(steps,[BLUE,OR]): ax.plot(st['z'],st['slope'],'o',color=c_,ms=5.5,zorder=5)
ax.text(0.2,-0.12,'step 1',color=BLUE,fontsize=9); ax.text(-2.95,-0.62,'step 2',color=OR,fontsize=9,ha='left')
ax.set_xlabel('margin $z$',fontsize=9.5); ax.set_ylabel("slope $L'(z)$",fontsize=9.5); ax.set_ylim(-1.05,0.1); grid(ax); ax.tick_params(labelsize=9); fig.tight_layout(); save(fig,'p_17A3.png')

# ---- 17A.4 optimal schedule, alpha = 1e-4
a_=1e-4; eta1=a_**-0.25; t0=1/(a_*eta1)-1; f=lambda t:1/(a_*(t+t0)); A['17A.4']={'eta1':round(eta1,4),'t0':round(t0,2),'eta_1000':round(f(1000),4),'eta_1e5':round(f(1e5),4),'check_rate':round(rate('optimal',1000,a_,0,0),4)}
fig,ax=small(2.4,1.7); tt=np.logspace(0,5,200); ax.loglog(tt,f(tt),color=GREEN,lw=1.6); ax.loglog([1,1000,1e5],[f(1),f(1000),f(1e5)],'o',color=PUR,ms=5,zorder=5)
ax.set_xlabel('update $t$',fontsize=9.5); ax.set_ylabel('$\\eta_t$',fontsize=9.5); ax.tick_params(labelsize=9); ax.grid(color=GRID,lw=0.5); fig.tight_layout(); save(fig,'p_17A4.png')

# ---- 17B.1 averaging (answer: the largest difference between a running mean and sklearn)
X,y=load_breast_cancer(return_X_y=True); Xs=StandardScaler().fit_transform(X); s=2*y-1.0; a=1e-3; n=len(y)
def run(epochs=3,sched='optimal',eta0=0.01):
    w=np.zeros(Xs.shape[1]); b=0.0; t=1; wb=np.zeros_like(w); bb=0.0; cnt=0
    for e in range(epochs):
        for i in range(n):
            eta=rate(sched,t,a,eta0,0.5); z=s[i]*(Xs[i]@w+b); g=s[i]*SLOPE['hinge'](z); w=(1-eta*a)*w-eta*g*Xs[i]; b=b-eta*g; t+=1; cnt+=1; wb+=(w-wb)/cnt; bb+=(b-bb)/cnt
    return w,b,wb,bb
w,b,wb,bb=run(); ref=SGDClassifier(alpha=a,average=True,max_iter=3,tol=None,shuffle=False).fit(Xs,y)
A['17B.1']={'max_dw_vs_sklearn':float(np.abs(wb-ref.coef_[0]).max()),'max_db':float(abs(bb-ref.intercept_[0])),'plain_vs_avg_norm':[round(float(np.linalg.norm(w)),4),round(float(np.linalg.norm(wb)),4)]}
fig,ax=small(2.4,1.7); ax.plot(np.arange(1,6),[1,0.5,0.8,0.55,0.62],color=GREY,lw=1.3,marker='o',ms=3.5,label='iterates'); ax.plot(np.arange(1,6),np.cumsum([1,0.5,0.8,0.55,0.62])/np.arange(1,6),color=GREEN,lw=1.6,marker='s',ms=3.5,label='running mean')
ax.set_xlabel('update $t$',fontsize=9.5); ax.set_ylabel('one weight',fontsize=9.5); ax.legend(fontsize=8.8,loc='upper right',borderaxespad=0.2); ax.tick_params(labelsize=9); grid(ax); ax.set_ylim(0.3,1.15); fig.tight_layout(); save(fig,'p_17B1.png')

# ---- 17B.2 stopping rules: n_iter_
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,stratify=y,random_state=0); sc=StandardScaler().fit(Xtr); Xa,Xb=sc.transform(Xtr),sc.transform(Xte); A['17B.2']={}
for es in (False,True):
    for tol in (1e-2,1e-3,1e-4):
        m=SGDClassifier(tol=tol,early_stopping=es,random_state=0,max_iter=1000).fit(Xa,ytr); A['17B.2'][f'early_stopping={es}, tol={tol:g}']={'n_iter':int(m.n_iter_),'test_acc':round(float(m.score(Xb,yte)),4)}
fig,ax=small(2.4,1.7); labs=['1e-2','1e-3','1e-4']; xs=np.arange(3); 
for es,c_,off in ((False,BLUE,-0.18),(True,PUR,0.18)):
    ax.bar(xs+off,[A['17B.2'][f'early_stopping={es}, tol={t}']['n_iter'] for t in ('0.01','0.001','0.0001')],width=0.34,color=c_,label=f'early_stopping={es}',hatch='' if not es else '//',edgecolor='white')
ax.set_xticks(xs); ax.set_xticklabels(labs,fontsize=9); ax.set_xlabel('tol',fontsize=9.5); ax.set_ylabel('epochs run',fontsize=9.5); ax.set_ylim(0,58); ax.legend(fontsize=8,loc='upper left',borderaxespad=0.2); ax.tick_params(labelsize=9); ax.grid(axis='y',color=GRID,lw=0.5); fig.tight_layout(); save(fig,'p_17B2.png')

# ---- 17B.3 multiclass one-versus-rest on iris
Xi,yi=load_iris(return_X_y=True); Xi=StandardScaler().fit_transform(Xi); mi=SGDClassifier(random_state=0,max_iter=1000).fit(Xi,yi); F=Xi@mi.coef_.T+mi.intercept_
A['17B.3']={'coef_shape':list(mi.coef_.shape),'agree_argmax':bool(np.array_equal(mi.classes_[F.argmax(1)],mi.predict(Xi))),'accuracy':round(float(mi.score(Xi,yi)),4)}
fig,ax=small(2.4,1.7)
for k,(c_,mk) in enumerate(zip([BLUE,OR,GREEN],['o','s','^'])): ax.scatter(Xi[yi==k,2],Xi[yi==k,3],s=10,color=c_,marker=mk,edgecolor='white',lw=0.4,zorder=4)
ax.set_xlabel('petal length',fontsize=9.5); ax.set_ylabel('petal width',fontsize=9.5); ax.tick_params(labelsize=9); grid(ax); fig.tight_layout(); save(fig,'p_17B3.png')

# ---- 17C.1 mistakes per epoch of the Perceptron on the four tiles (epochs 1..8)
w,b,log=prun([0,1,2,3]); per=[sum(1 for e,_ in log if e==k) for k in range(1,9)]; A['17C.1']={'mistakes_per_epoch':per}
fig,ax=small(2.4,1.7); ax.bar(range(1,9),per,color=[RED if v>0 else GRID for v in per],width=0.7); ax.set_xlabel('epoch',fontsize=9.5); ax.set_ylabel('mistakes',fontsize=9.5); ax.set_yticks([0,1,2,3]); ax.tick_params(labelsize=9); ax.grid(axis='y',color=GRID,lw=0.5)
ax.annotate('stops here\n($n$_iter_ = 8)',xy=(8,0.05),xytext=(4.3,2.3),fontsize=8.8,color=NAVY,arrowprops=dict(arrowstyle='-|>',color=NAVY,lw=0.8)); fig.tight_layout(); save(fig,'p_17C1.png')

# ---- 17C.2 overlapping classes: the perceptron never settles
Xo,yo=make_blobs(n_samples=100,centers=[(-1.0,-0.7),(1.0,0.7)],cluster_std=1.3,random_state=3); m=Perceptron(random_state=0,tol=None,max_iter=1); acc=[]
for k in range(40): m.partial_fit(Xo,yo,classes=[0,1]); acc.append(m.score(Xo,yo))
A['17C.2']={'min':round(float(min(acc)),3),'max':round(float(max(acc)),3),'last':round(float(acc[-1]),3)}
fig,ax=small(2.4,1.7); ax.plot(range(1,41),acc,color=BLUE,lw=1.3); ax.set_xlabel('epoch',fontsize=9.5); ax.set_ylabel('training accuracy',fontsize=9.5); ax.tick_params(labelsize=9); grid(ax); ax.set_ylim(0.5,1.0); fig.tight_layout(); save(fig,'p_17C2.png')

# ---- 17C.3 squared against hinge on the right side
z=np.linspace(-1.5,3,300); fig,ax=small(2.4,1.7); ax.plot(z,(1-z)**2,color=RED,lw=1.6,label='squared  $(1-z)^2$'); ax.plot(z,np.maximum(0,1-z),color=BLUE,lw=1.6,ls='--',label='hinge')
ax.axvspan(1,3,color=SOFT if False else '#FDF0E0',lw=0); ax.text(1.6,2.1,'easy\nsamples',fontsize=9,color=OR,ha='center'); ax.set_ylim(-0.1,5); ax.set_xlim(-1.5,3); ax.set_xlabel('margin $z$',fontsize=9.5); ax.set_ylabel('loss',fontsize=9.5)
ax.legend(fontsize=8.3,loc='upper left',borderaxespad=0.2); ax.tick_params(labelsize=9); grid(ax); fig.tight_layout(); save(fig,'p_17C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

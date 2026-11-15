# Problem figures of Chapter 23. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from scipy.optimize import minimize
from svr_scratch import MySVR
plt.rcParams.update({'xtick.labelsize':10.6,'ytick.labelsize':10.6,'axes.labelsize':10.6,'legend.fontsize':10.6})
A={}; CHK={}
GL='#DDF0E4'
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    import matplotlib.text as mtext
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image; w,h=Image.open(name).size; pw=min(1.55*110,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
Xd=np.array([[1.],[2.],[3.],[4.],[5.]]); yd=np.array([3.,4.,6.,7.,10.])
# ---- 23A.1 the loss of five residuals
r=np.array([-1.5,-0.4,0.2,0.9,2.1]); eps=0.5; L=np.maximum(0,np.abs(r)-eps); C=2.0
A['23A.1']={'L':[round(float(v),2) for v in L],'sumL':round(float(L.sum()),2),'sum_abs':round(float(np.abs(r).sum()),2),'sum_sq':round(float((r**2).sum()),2),'inside':int((np.abs(r)<=eps).sum()),'C_sumL':round(float(C*L.sum()),2),'J_w4':round(float(0.5*4+C*L.sum()),2)}
fig,ax=small(); rr=np.linspace(-2.6,2.6,200); ax.axvspan(-eps,eps,color='#E8DDF2',lw=0); ax.plot(rr,np.maximum(0,np.abs(rr)-eps),color=RED,lw=1.8); ax.scatter(r,L,s=24,color=NAVY,zorder=5)
for k,(a_,b_) in enumerate(zip(r,L)): ax.text(a_,b_+0.2,str(k+1),ha='center',fontsize=10.6,color=NAVY)
ax.set_xlabel('$r$'); ax.set_ylabel('$L_{\\varepsilon}(r)$'); ax.set_xlim(-2.6,2.6); ax.set_ylim(-0.1,2.3); sv(fig,'p_23A1.png')
# ---- 23A.2 two samples
X2=np.array([[1.],[3.]]); y2=np.array([1.,4.]); m=SVR(kernel='linear',C=1.0,epsilon=0.5,tol=1e-10).fit(X2,y2)
K2=X2@X2.T; bv=m.dual_coef_[0]; dual=-0.5*bv@K2@bv+y2@bv-0.5*np.abs(bv).sum(); prim=0.5*float(m.coef_[0,0]**2)
A['23A.2']={'w':round(float(m.coef_[0,0]),4),'b':round(float(m.intercept_[0]),4),'beta':[round(float(v),4) for v in bv],'support':[int(v) for v in m.support_],'f2':round(float(m.predict([[2.]])[0]),4),'primal':round(prim,4),'dual':round(float(dual),4)}
fig,ax=small(); xx=np.array([0.2,3.8]); f=m.predict(xx[:,None]); ax.fill_between(xx,f-0.5,f+0.5,color=GL,lw=0); ax.plot(xx,f,color=GREEN,lw=1.8); ax.scatter(X2[:,0],y2,s=30,color=BLUE,zorder=5,edgecolor='white')
ax.text(1.1,0.55,'(1, 1)',fontsize=10.6,color=NAVY); ax.text(2.0,4.35,'(3, 4)',fontsize=10.6,color=NAVY); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xlim(0.2,3.8); ax.set_ylim(0,5.2); sv(fig,'p_23A2.png')
# ---- 23A.3 sorting six samples
r3=np.array([0.1,-0.4,0.9,-1.2,0.4,0.0]); e3=0.4; C3=2.0; xi=np.maximum(0,np.abs(r3)-e3)
A['23A.3']={'xi':[round(float(v),2) for v in xi],'sum_xi':round(float(xi.sum()),2),'J':round(float(0.5*2.25+C3*xi.sum()),3)}
fig,ax=small(); ax.axvspan(-e3,e3,color='#E8DDF2',lw=0); ax.axhline(0,color=GRID,lw=0.8)
for k,a_ in enumerate(r3): ax.scatter(a_,0,s=26,color=BLUE,zorder=5,edgecolor='white'); ax.text(a_,0.16+0.0*k+(0.22 if k in (0,) else 0),str(k+1),ha='center',fontsize=10.6,color=NAVY)
ax.text(0,-0.45,'$\\pm\\varepsilon$',ha='center',fontsize=10.6,color=PUR); ax.set_xlim(-1.5,1.2); ax.set_ylim(-0.7,0.8); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('residual  $r$'); sv(fig,'p_23A3.png')
# ---- 23A.4 nu arithmetic
n,nu,C4=50,0.3,2.0; A['23A.4']={'max_outside':int(nu*n),'min_sv':int(nu*n),'sum_beta':nu*n*C4,'from_12_bound':12*C4,'rest':nu*n*C4-12*C4,'min_free':int(np.ceil((nu*n*C4-12*C4)/C4+1e-9)) if ((nu*n*C4-12*C4)/C4)%1 else int((nu*n*C4-12*C4)/C4)+1,'min_sv_given_12':12+4}
fig,ax=small(); k=0
for i in range(5):
    for j in range(10):
        col='#9DB8DE'; ec='none'
        if k<12: col=BLUE; ec=RED
        elif k<16: col=BLUE; ec=NAVY
        ax.scatter(j,-i,s=70,facecolor=col,edgecolor=ec,lw=1.3); k+=1
ax.set_xlim(-0.6,9.6); ax.set_ylim(-4.6,0.6); ax.axis('off'); ax.text(4.5,-5.0,'50 samples',ha='center',fontsize=10.6,color=NAVY); ax.set_ylim(-5.4,0.6); sv(fig,'p_23A4.png')
# ---- 23B.1 scale of y
s=60.0; m1=SVR(kernel='linear',C=1,epsilon=1,tol=1e-10).fit(Xd,yd); m2=SVR(kernel='linear',C=1,epsilon=1,tol=1e-10).fit(Xd,s*yd); m3=SVR(kernel='linear',C=s,epsilon=s,tol=1e-10).fit(Xd,s*yd)
A['23B.1']={'minutes':{'w':round(float(m1.coef_[0,0]),4),'b':round(float(m1.intercept_[0]),4),'f6':round(float(m1.predict([[6.]])[0]),4)},'seconds_default':{'w':round(float(m2.coef_[0,0]),4),'b':round(float(m2.intercept_[0]),4),'f6':round(float(m2.predict([[6.]])[0]),3)},
  'seconds_scaled':{'w':round(float(m3.coef_[0,0]),4),'b':round(float(m3.intercept_[0]),4),'f6':round(float(m3.predict([[6.]])[0]),3),'max_diff_vs_60x':float(np.abs(m3.predict(Xd)-s*m1.predict(Xd)).max())}}
fig,ax=small(); xx=np.linspace(0.5,5.5,20)[:,None]; ax.scatter(Xd[:,0],s*yd,s=24,color=BLUE,zorder=5,edgecolor='white'); ax.plot(xx[:,0],m2.predict(xx),color=RED,lw=1.5,label='$C=1$, $\\varepsilon=1$'); ax.plot(xx[:,0],m3.predict(xx),color=GREEN,lw=1.8,label='$C=60$, $\\varepsilon=60$')
ax.set_xlabel('$x$ (km)'); ax.set_ylabel('$y$ (s)'); ax.legend(loc='upper left',fontsize=10.6,handlelength=1.4,borderaxespad=0.2); ax.set_ylim(0,700); sv(fig,'p_23B1.png')
# ---- 23B.3 sample weights in MySVR
class WSVR(MySVR):
    def fit(self,X,y,sw):
        n,C,eps=len(y),self.C,self.epsilon; K=self._k(X,X); t=1e-6*C; Cw=C*sw
        def nd(a):
            be=a[:n]-a[n:]; g=K@be; return 0.5*be@g+eps*a.sum()-y@be, np.r_[g+eps-y,-g+eps+y]
        eq={'type':'eq','fun':lambda a:a[:n].sum()-a[n:].sum()}
        res=minimize(nd,np.zeros(2*n),jac=True,method='SLSQP',constraints=eq,bounds=[(0,c) for c in Cw]*2 if False else [(0,c) for c in np.r_[Cw,Cw]],tol=1e-12)
        al,als=res.x[:n],res.x[n:]; be=al-als; s_=K@be
        up=(al>t*sw)&(al<Cw-t*sw); lo=(als>t*sw)&(als<Cw-t*sw); b=np.r_[(y-eps-s_)[up],(y+eps-s_)[lo]]
        sv_=np.abs(be)>t; self.support_=np.where(sv_)[0]; self.dual_coef_=be[sv_]; self.support_vectors_=X[sv_]; self.intercept_=b.mean(); self.beta_=be; return self
rg=np.random.default_rng(0); Xb=rg.normal(size=(40,3)); yb=np.sin(Xb[:,0])+0.5*Xb[:,1]**2+0.1*rg.normal(size=40); sw=np.ones(40); sw[:10]=0.25; Xt=rg.normal(size=(200,3))
a=WSVR(C=1.0,epsilon=0.1,kernel='rbf',gamma=0.5).fit(Xb,yb,sw); ref=SVR(C=1.0,epsilon=0.1,kernel='rbf',gamma=0.5,tol=1e-8).fit(Xb,yb,sample_weight=sw)
dd=float(np.abs(a.predict(Xt)-ref.predict(Xt)).max()); plain=SVR(C=1.0,epsilon=0.1,kernel='rbf',gamma=0.5,tol=1e-8).fit(Xb,yb)
bref=np.zeros(40); bref[ref.support_]=ref.dual_coef_[0]; bpl=np.zeros(40); bpl[plain.support_]=plain.dual_coef_[0]
A['23B.3']={'max_diff':dd,'same_support':bool(np.array_equal(a.support_,ref.support_)),'n_sv_weighted':int(len(ref.support_)),'n_sv_plain':int(len(plain.support_)),'max_abs_beta_first10':round(float(np.abs(bref[:10]).max()),4),'max_abs_beta_rest':round(float(np.abs(bref[10:]).max()),4),'bound_first10':0.25,'max_abs_beta_plain':round(float(np.abs(bpl).max()),4)}
fig,ax=small(); ax.bar(np.arange(40),bref,color=np.where(np.arange(40)<10,PUR,BLUE),width=0.8); ax.axhline(0.25,color=PUR,lw=0.8,ls=':'); ax.axhline(-0.25,color=PUR,lw=0.8,ls=':'); ax.axhline(1,color=GREY,lw=0.8,ls=':'); ax.axhline(-1,color=GREY,lw=0.8,ls=':')
ax.set_xlabel('sample $i$'); ax.set_ylabel('$\\beta_i$'); ax.set_xlim(-1,40); ax.set_ylim(-1.25,1.25); sv(fig,'p_23B2.png')
# ---- 23B.2 epsilon and sparsity on 300 samples
rg=np.random.default_rng(1); xa=rg.uniform(0,6,300); ya=np.sin(xa)+0.25*rg.normal(size=300); Xa=xa[:,None]; Xtr,ytr,Xte,yte=Xa[:200],ya[:200],Xa[200:],ya[200:]
es=[0,0.1,0.2,0.3,0.5,0.8]; rows=[]
for e in es:
    m=SVR(C=10.0,epsilon=e,gamma=0.5).fit(Xtr,ytr); rows.append((e,len(m.support_),float(np.sqrt(np.mean((m.predict(Xte)-yte)**2)))))
k=KernelRidge(alpha=0.1,kernel='rbf',gamma=0.5).fit(Xtr,ytr-ytr.mean()); krr=float(np.sqrt(np.mean((k.predict(Xte)+ytr.mean()-yte)**2)))
A['23B.2']={'rows':[[r_[0],r_[1],round(r_[2],4)] for r_ in rows],'krr_rmse':round(krr,4),'noise_sd':0.25}
fig,ax=small(1.8,1.75); ax.plot([r_[0] for r_ in rows],[r_[1]/200 for r_ in rows],'-o',color=NAVY,ms=3.5,lw=1.4); ax.set_xlabel('$\\varepsilon$'); ax.set_ylabel('share of SVs',color=NAVY); ax.set_ylim(0,1.05)
ax2=ax.twinx(); ax2.plot([r_[0] for r_ in rows],[r_[2] for r_ in rows],'-s',color=RED,ms=3.5,lw=1.4); ax2.set_ylabel('test RMSE',color=RED); ax2.spines['right'].set_visible(True); ax2.set_ylim(0.2,0.45); ax2.tick_params(labelsize=10.6)
sv(fig,'p_23B3.png')
# ---- 23C.1 which fits change
fig,ax=small(2.4,1.75); xs5=np.array([1,2,3,4,5.]); ys5=np.array([2.0,2.4,3.6,3.8,5.9]); mm=SVR(kernel='linear',C=10,epsilon=0.5,tol=1e-10).fit(xs5[:,None],ys5); xx=np.linspace(0.5,5.5,10); f=mm.predict(xx[:,None])
ax.fill_between(xx,f-0.5,f+0.5,color=GL,lw=0); ax.plot(xx,f,color=GREEN,lw=1.6); ax.scatter(xs5,ys5,s=24,color=BLUE,zorder=5,edgecolor='white')
for k_,(a_,b_) in enumerate(zip(xs5,ys5)): ax.text(a_+0.1,b_-0.35,'ABCDE'[k_],fontsize=10.6,color=NAVY)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xlim(0.5,5.5); ax.set_ylim(1,7); sv(fig,'p_23C1.png')
A['23C.1']={'fit':{'w':round(float(mm.coef_[0,0]),4),'b':round(float(mm.intercept_[0]),4),'support':[int(v) for v in mm.support_],'residuals':[round(float(v),3) for v in ys5-mm.predict(xs5[:,None])]}}
# ---- 23C.2 tube swallows the data
fig,ax=small(); xx=np.linspace(0.5,5.5,10)
for e,col,ls in ((1.0,GREEN,'-'),(2.0,OR,'--'),(3.5,RED,':')):
    mm=SVR(kernel='linear',C=10,epsilon=e,tol=1e-10).fit(Xd,yd); ax.plot(xx,mm.predict(xx[:,None]),color=col,lw=1.6,ls=ls,label='$\\varepsilon=%g$'%e)
ax.scatter(Xd[:,0],yd,s=24,color=BLUE,zorder=5,edgecolor='white'); ax.set_xlabel('$x$ (km)'); ax.set_ylabel('$y$ (min)'); ax.legend(loc='upper left',fontsize=10.6,handlelength=1.8,borderaxespad=0.2); ax.set_ylim(1,11.5); sv(fig,'p_23C2.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print(json.dumps(A,indent=0)[:3000]); print(CHK)

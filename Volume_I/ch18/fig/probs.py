# Problem figures of Chapter 18. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.datasets import load_wine
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def sv(fig,name,printed_px=135):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w,h=Image.open(name).size; w/=240; h/=240
    pr=min(1.55*printed_px,235*w/h)/96
    CHK[name]={'fig_in':round(w,2),'printed_in':round(pr,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*pr/w,1)}
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
npdf=lambda x,m,v:np.exp(-(x-m)**2/(2*v))/np.sqrt(2*np.pi*v)
# ---- 18A.1
m0,m1,v,p0,p1=2.0,6.0,2.0,0.7,0.3
xs_=m0/2+m1/2+v/(m1-m0)*np.log(p0/p1); w_=(m1-m0)/v; z=w_*(4.5-xs_); A['18A.1']={'x_star':round(float(xs_),4),'x_mid':4.0,'P1_at_4.5':round(float(1/(1+np.exp(-z))),4),'w':w_}
fig,ax=small(3.0,1.25); xx=np.linspace(-1.5,9.5,400); ax.plot(xx,p0*npdf(xx,m0,v),color=BLUE,lw=1.5); ax.plot(xx,p1*npdf(xx,m1,v),color=OR,lw=1.5,ls='--')
ax.axvline(4,color=GREY,lw=1,ls=':'); ax.axvline(xs_,color=GREEN,lw=1.6); ax.set_yticks([]); ax.set_xlabel('$x$'); ax.set_ylabel('$\\pi_k\\,p(x\\mid k)$'); ax.text(0.2,0.215,'class 0',color=BLUE,fontsize=9,ha='center'); ax.text(7.6,0.13,'class 1',color=OR,fontsize=9,ha='center')
ax.text(xs_+0.2,0.19,'$x^*$',color=GREEN,fontsize=9.5); ax.set_ylim(0,0.26); ax.set_xlim(-1.5,9.5); sv(fig,'p_18A1.png')
# ---- 18A.2
S=np.diag([2.0,0.5]); mu0=np.array([0.,0.]); mu1=np.array([4.,2.]); pt=np.array([2.5,0.5]); Si=np.linalg.inv(S)
dM=lambda m:float((pt-m)@Si@(pt-m)); eu=lambda m:float(np.linalg.norm(pt-m)); wv=Si@(mu1-mu0); bb=-0.5*(mu0+mu1)@wv; sc_=float(wv@pt+bb)
A['18A.2']={'euclid':[round(eu(mu0),4),round(eu(mu1),4)],'maha2':[dM(mu0),dM(mu1)],'w':wv.tolist(),'b':float(bb),'score':sc_,'P1':round(float(1/(1+np.exp(-sc_))),4)}
fig,ax=small(2.4,1.75)
for m,c in ((mu0,BLUE),(mu1,OR)): ax.add_patch(Ellipse(m,2*2*np.sqrt(2.0),2*2*np.sqrt(0.5),fill=False,ec=c,lw=1.3)); ax.plot(*m,'+',color=c,ms=8,mew=1.5)
for m in (mu0,mu1): ax.plot(*np.c_[pt,m],color=GREY,lw=0.9,ls=':')
ax.plot(*pt,'x',color=RED,ms=7,mew=1.8,zorder=6); ax.text(2.35,1.0,'$\\mathbf{x}$',color=RED,fontsize=10)
ax.set_xlim(-3.4,7.4); ax.set_ylim(-2.2,3.8); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); sv(fig,'p_18A2.png')
# ---- 18A.3
K,pp=4,20; cl=lambda K,p:K*p+p*(p+1)//2+K-1; cq=lambda K,p:K*p+K*p*(p+1)//2+K-1
A['18A.3']={'LDA':cl(K,pp),'QDA':cq(K,pp),'K':K,'p':pp,'per_class':75,'p_big':80,'LDA_p80':cl(K,80),'QDA_p80':cq(K,80)}
fig,ax=small(); ps=np.arange(2,41); ax.plot(ps,[cl(K,p) for p in ps],color=BLUE,lw=1.6,label='LDA'); ax.plot(ps,[cq(K,p) for p in ps],color=OR,lw=1.6,ls='--',label='QDA')
ax.plot([pp,pp],[cl(K,pp),cq(K,pp)],'o',color=GREEN,ms=4); ax.set_yscale('log'); ax.set_xlabel('features $p$  ($K$ = 4)'); ax.set_ylabel('parameters'); ax.legend(loc='upper left',fontsize=9); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); sv(fig,'p_18A3.png')
# ---- 18B.1
Xw,yw=load_wine(return_X_y=True); lw=LDA().fit(Xw,yw); Z=lw.transform(Xw); Wc=np.vstack([Z[yw==k]-Z[yw==k].mean(0) for k in range(3)])
A['18B.1']={'within_var':np.round(np.diag(np.cov(Wc.T,bias=True)),4).tolist(),'within_cov_offdiag':round(float(np.cov(Wc.T,bias=True)[0,1]),6),'centroids_abs':np.round(np.abs([Z[yw==k].mean(0) for k in range(3)]),3).tolist(),'ratio':np.round(lw.explained_variance_ratio_,4).tolist()}
fig,ax=small(2.4,1.6)
rg=np.random.default_rng(0)
for k in range(3): ax.scatter(Z[yw==k,0],rg.uniform(-0.3,0.3,(yw==k).sum())+k,s=9,color=[BLUE,OR,TEAL][k],marker=['o','s','^'][k],edgecolor='white',lw=0.3)
ax.set_yticks([0,1,2]); ax.set_yticklabels(['class 0','class 1','class 2']); ax.set_xlabel('first discriminant'); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_ylim(-0.6,2.6); sv(fig,'p_18B1.png')
# ---- 18B.2
rg=np.random.default_rng(0); P_,N_=50,40; idx=np.arange(P_); L=np.linalg.cholesky(0.6**np.abs(idx[:,None]-idx[None,:])); dl=0.3*(-1.0)**idx
yy=np.arange(N_)%2; Xd=rg.normal(size=(N_,P_))@L.T+yy[:,None]*dl
Sw=sum(0.5*np.cov(Xd[yy==k].T,bias=True) for k in (0,1)); tr=np.trace(Sw)/P_
ev=np.sort(np.linalg.eigvalsh(Sw))[::-1]; rk=int(np.sum(ev>1e-10*ev[0]))
def sh(a): return (1-a)*Sw+a*tr*np.eye(P_)
A['18B.2']={'rank':rk,'trace_over_p':round(float(tr),4),'ev_min_0':float(ev[-1]),'ev_min_0.1':round(float(np.linalg.eigvalsh(sh(0.1))[0]),4),'ev_min_0.5':round(float(np.linalg.eigvalsh(sh(0.5))[0]),4),
  'cond_0.1':round(float(np.linalg.cond(sh(0.1))),1),'cond_0.5':round(float(np.linalg.cond(sh(0.5))),1),'ev_max':round(float(ev[0]),4)}
fig,ax=small(); ax.semilogy(np.arange(1,P_+1),np.maximum(ev,1e-17),color=RED,lw=1.4,label='$a$ = 0'); ax.semilogy(np.arange(1,P_+1),np.sort(np.linalg.eigvalsh(sh(0.5)))[::-1],color=PUR,lw=1.4,ls='--',label='$a$ = 0.5')
ax.set_ylim(1e-6,20); ax.set_xlabel('index of eigenvalue'); ax.set_ylabel('eigenvalue of $\\Sigma_a$'); ax.legend(loc='lower left',fontsize=9); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); sv(fig,'p_18B2.png')
# ---- 18C.1
rg=np.random.default_rng(3); X0=rg.multivariate_normal([0,0],[[1,0.3],[0.3,0.6]],40); X1=rg.multivariate_normal([3,1.5],[[1,0.3],[0.3,0.6]],40); out=np.array([[7.5,-3.0]])
Xc=np.r_[X0,X1]; yc=np.r_[np.zeros(40),np.ones(40)].astype(int); Xo=np.r_[Xc,out]; yo=np.r_[yc,[0]]
l1=LDA().fit(Xc,yc); l2=LDA().fit(Xo,yo)
A['18C.1']=None; A['18C.2']=None
fig,ax=small(2.4,1.8); ax.scatter(X0[:,0],X0[:,1],s=9,color=BLUE,edgecolor='white',lw=0.3); ax.scatter(X1[:,0],X1[:,1],s=9,color=OR,marker='s',edgecolor='white',lw=0.3); ax.scatter(*out.T,s=26,color=BLUE,edgecolor=RED,lw=1.3,zorder=6)
ax.set_xlim(-3,9); ax.set_ylim(-4,4); gx,gy=np.meshgrid(np.linspace(-3,9,200),np.linspace(-4,4,200)); G=np.c_[gx.ravel(),gy.ravel()]
ax.contour(gx,gy,l1.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=1.5,linestyles='--'); ax.contour(gx,gy,l2.decision_function(G).reshape(gx.shape),levels=[0],colors=[GREEN],linewidths=1.5)
ax.text(7.2,-2.4,'outlier',color=RED,fontsize=9,ha='center'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); sv(fig,'p_18C1.png')
# ---- 18C.2
fig,ax=new(2.4,1.7); T3=np.array([[0.2,0.2],[2.6,0.4],[1.3,1.9]]); mb=T3.mean(0)
ax.add_patch(Polygon(T3,fc=LB,ec=BLUE,lw=0.9,alpha=0.8))
for pnt,c,l,off in zip(T3,[BLUE,OR,TEAL],['$\\mu_1$','$\\mu_2$','$\\mu_3$'],[(-0.25,-0.1),(0.28,-0.05),(0.0,0.17)]): ax.plot(*pnt,'o',color=c,ms=6); ax.text(pnt[0]+off[0],pnt[1]+off[1],l,fontsize=10,color=c,ha='center')
ax.plot(*mb,'+',color=NAVY,ms=8,mew=1.5); ax.text(mb[0]+0.08,mb[1]-0.05,'$\\bar\\mu$',fontsize=10,color=NAVY,va='top')
ax.set_xlim(-0.2,3.2); ax.set_ylim(0.0,2.55); sv(fig,'p_18C2.png')
# ---- 18A.4 Fisher's criterion along the axes (data of Example 18.6-3)
X8=np.array([[1,1],[3,3],[1,2],[3,2],[4,2],[6,4],[5,2],[5,4]],float); y8=np.r_[0,0,0,0,1,1,1,1]
S8=sum(0.5*np.cov(X8[y8==k].T,bias=True) for k in (0,1)); d8=X8[y8==1].mean(0)-X8[y8==0].mean(0)
Jf=lambda a:float((a@d8)**2/(a@S8@a)); w8=np.linalg.solve(S8,d8)
A['18A.4']={'Sigma':S8.tolist(),'J_x1':round(Jf(np.array([1.,0])),4),'J_x2':round(Jf(np.array([0.,1])),4),'J_fisher':round(Jf(w8),4),'J_meandiff':round(Jf(d8),4)}
fig,ax=small(2.4,1.8); ax.scatter(X8[y8==0,0],X8[y8==0,1],s=22,color=BLUE,edgecolor='white',zorder=5); ax.scatter(X8[y8==1,0],X8[y8==1,1],s=22,color=OR,marker='s',edgecolor='white',zorder=5)
c8=X8.mean(0)
for a,c,ls,l in ((np.array([1.,0]),GREY,'--','$a$=(1,0)'),(np.array([0.,1]),PUR,':','$a$=(0,1)'),(w8/np.linalg.norm(w8),GREEN,'-','Fisher')):
    ax.plot([c8[0]-2.6*a[0],c8[0]+2.6*a[0]],[c8[1]-2.6*a[1],c8[1]+2.6*a[1]],color=c,ls=ls,lw=1.4,label=l)
ax.set_xlim(0,7); ax.set_ylim(-0.3,5.3); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.legend(loc='upper left',fontsize=8,handlelength=1.6,borderaxespad=0.1,labelspacing=0.2); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); sv(fig,'p_18A4.png')
# ---- 18B.3 priors as a lever (case-study split of section 18.11)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
Xb,yb=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=0.25,random_state=0,stratify=yb)
pm=[float(np.mean(ytr==0)),0.7,0.9,0.99]; miss=[]; fa=[]; ic=[]
for p0 in pm:
    m=LDA(priors=[p0,1-p0]).fit(Xtr,ytr); cm=confusion_matrix(yte,m.predict(Xte)); miss.append(int(cm[0,1])); fa.append(int(cm[1,0])); ic.append(round(float(m.intercept_[0]),3))
A['18B.3']={'prior_malignant':[round(v,4) for v in pm],'missed_malignant':miss,'false_alarms':fa,'intercept':ic}
fig,ax=small(2.4,1.75); xb=np.arange(4); ax.bar(xb-0.18,miss,0.36,color=RED,label='missed malignant'); ax.bar(xb+0.18,fa,0.36,color=GREY,hatch='//',edgecolor='white',label='false alarms')
for k in range(4): ax.text(xb[k]-0.18,miss[k]+0.4,str(miss[k]),ha='center',fontsize=8.5,color=NAVY); ax.text(xb[k]+0.18,fa[k]+0.4,str(fa[k]),ha='center',fontsize=8.5,color=NAVY)
ax.set_xticks(xb); ax.set_xticklabels(['%.2f'%pm[0],'0.7','0.9','0.99']); ax.set_xlabel('prior of malignant'); ax.set_ylabel('test tumours'); ax.set_ylim(0,25); ax.legend(loc='upper left',fontsize=8.5,borderaxespad=0.1); sv(fig,'p_18B3.png')
# ---- 18C.3 class sizes against the number of features
fig,ax=small(2.4,1.7); nk=[30,30,8]; ax.bar([0,1,2],nk,0.55,color=[BLUE,OR,TEAL]); ax.axhline(12,color=RED,lw=1.4,ls='--'); ax.text(2.45,13.0,'$p$ = 12',color=RED,fontsize=9,ha='right',va='bottom')
for k,v in enumerate(nk): ax.text(k,v+0.8,str(v),ha='center',fontsize=9,color=NAVY)
ax.set_xticks([0,1,2]); ax.set_xticklabels(['class 0','class 1','class 2']); ax.set_ylabel('samples $n_k$'); ax.set_ylim(0,36); sv(fig,'p_18C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1); print(json.dumps(A)); print(json.dumps(CHK))

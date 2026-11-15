# Problem figures of Chapter 46. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, MinCovDet, GraphicalLasso, empirical_covariance)
from sklearn.preprocessing import StandardScaler
from returns import true_model, draw
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.4): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def ellipse(ax,loc,cov,c2,**kw):
    lam,V=np.linalg.eigh(cov); t=np.linspace(0,2*np.pi,200); P=np.sqrt(c2)*(V*np.sqrt(lam))@np.vstack([np.cos(t),np.sin(t)]); ax.plot(loc[0]+P[0],loc[1]+P[1],**kw)
Sigma,Theta,sd=true_model(); p=20
# 46A.1
S=np.array([[4,2],[2,3.]]); Th=np.linalg.inv(S); pts=np.array([[2,-1],[2,1.]]); d2=[float(q@Th@q) for q in pts]
A['46A.1']={'det':float(np.linalg.det(S)),'precision_times_8':(Th*8).round(4).tolist(),'d2':d2,'euclid2':[float(q@q) for q in pts]}
fig,ax=small(2.4,1.5); ellipse(ax,[0,0],S,4,color=GREEN,lw=1.4)
for q,nm in zip(pts,'AB'): ax.plot(*q,'s',color=RED,ms=5); ax.text(q[0]+0.15,q[1]-0.1,nm,color=RED,fontsize=9,weight='bold')
ax.plot(0,0,'+',color=NAVY); ax.set_aspect('equal'); ax.set_xlim(-3,3.2); ax.set_ylim(-3,3); ax.set_xlabel('$x_1-\\bar{x}_1$'); ax.set_ylabel('$x_2-\\bar{x}_2$'); grid(ax); sv(fig,'p_46A1.png')
# 46A.2
S2=np.array([[5,4],[4,5.]]); ev=np.linalg.eigvalsh(S2); a=0.25; S2a=(1-a)*S2+a*5*np.eye(2); eva=np.linalg.eigvalsh(S2a)
A['46A.2']={'eig':ev.tolist(),'shrunk':S2a.tolist(),'eig_shrunk':eva.tolist(),'cond':[float(ev[1]/ev[0]),float(eva[1]/eva[0])]}
fig,ax=small(2.4,1.4); xx=np.arange(2); ax.bar(xx-0.18,ev[::-1],0.34,color=BLUE,label='$\\alpha=0$'); ax.bar(xx+0.18,eva[::-1],0.34,color=GREEN,label='$\\alpha=0.25$'); ax.set_xticks(xx); ax.set_xticklabels(['$\\lambda_1$','$\\lambda_2$']); ax.set_ylim(0,12.5); ax.legend(loc='upper right',handlelength=1,fontsize=8.5); sv(fig,'p_46A2.png')
# 46A.3
T3=np.array([[2,-1,0],[-1,2,-1],[0,-1,2.]]); S3=np.linalg.inv(T3); r13=S3[0,2]/np.sqrt(S3[0,0]*S3[2,2]); r12=S3[0,1]/np.sqrt(S3[0,0]*S3[1,1])
A['46A.3']={'Sigma':S3.round(4).tolist(),'partial12':float(-T3[0,1]/2),'partial13':0.0,'corr12':float(r12),'corr13':float(r13)}
fig,ax=small(2.4,1.2); ax.axis('off'); ax.set_xlim(0,3); ax.set_ylim(0,1.5)
for k,(x,nm) in enumerate(((0.5,'1'),(1.5,'2'),(2.5,'3'))): ax.plot(x,0.7,'o',color=BLUE,ms=14); ax.text(x,0.7,nm,color='white',ha='center',va='center',fontsize=9,weight='bold')
ax.plot([0.5,1.5],[0.7,0.7],color=GREEN,lw=2,zorder=0); ax.plot([1.5,2.5],[0.7,0.7],color=GREEN,lw=2,zorder=0); ax.text(1.5,1.2,'$\\Theta_{13}=0$: no edge',ha='center',fontsize=9,color=NAVY); sv(fig,'p_46A3.png')
# 46A.4
S4=np.array([[1,0.6,0.2],[0.6,1,0.5],[0.2,0.5,1.]]); am=float(np.max(np.abs(S4-np.eye(3))))
g=GraphicalLasso(alpha=am,enet_tol=1e-6).fit(S4) if False else None
A['46A.4']={'alpha_max':am}
fig,ax=small(2.4,1.4); v=[0.6,0.2,0.5]; ax.bar(['$S_{12}$','$S_{13}$','$S_{23}$'],v,color=BLUE,width=0.5); ax.axhline(am,color=RED,ls='--',lw=1.2); ax.set_ylim(0,0.8); ax.set_ylabel('$|S_{ij}|$'); grid(ax); sv(fig,'p_46A4.png')
# 46B.1 rank and score against n
res=[]; Xte=draw(500,99)
for n in (10,15,19,20,21,25,30,40):
    X=draw(n,0); e=EmpiricalCovariance().fit(X); lw=LedoitWolf().fit(X); res.append((n,int(np.linalg.matrix_rank(e.covariance_)),float(e.score(Xte)),float(lw.score(Xte)),float(lw.shrinkage_)))
A['46B.1']=res
fig,ax=small(2.6,1.4); ax.plot([r[0] for r in res],[r[1] for r in res],'o-',color=BLUE,ms=3.5); ax.plot([10,40],[10,40],':',color=GREY); ax.axhline(20,color=RED,ls='--',lw=1); ax.set_xlabel('days $n$'); ax.set_ylabel('rank of $\\mathbf{S}$'); grid(ax); sv(fig,'p_46B1.png')
# 46B.2 edges against alpha / alpha_max
from sklearn.covariance._graph_lasso import alpha_max
Z=StandardScaler().fit_transform(draw(150,1)); amz=alpha_max(empirical_covariance(Z)); fr=(0.8,0.6,0.4,0.2,0.1); ed=[]
for f in fr:
    P=GraphicalLasso(alpha=f*amz,enet_tol=1e-6).fit(Z).precision_; ed.append(int(((np.abs(P)>1e-10).sum()-p)//2))
A['46B.2']={'alpha_max':float(amz),'fractions':fr,'edges':ed}
fig,ax=small(2.4,1.4); ax.plot(fr,ed,'o-',color=PUR,ms=3.5); ax.axhline(18,color=GREEN,ls=':',lw=1.2); ax.set_xlabel('$\\alpha/\\alpha_{\\max}$'); ax.set_ylabel('edges'); grid(ax); sv(fig,'p_46B2.png')
# 46B.3 number injected
wine=load_wine(); yw=wine.target; Xw=wine.data[:,[0,6,9,12]]; cut=chi2(4).isf(0.025); rb=[]
for k in (4,8,12,16,20):
    bad=np.random.RandomState(0).choice(np.where(yw==2)[0],k,replace=False); Wd=np.vstack([Xw[yw==0],Xw[bad]]); ib=np.r_[np.zeros(59,bool),np.ones(k,bool)]
    fe=EmpiricalCovariance().fit(Wd).mahalanobis(Wd)>cut; fm=MinCovDet(random_state=0).fit(Wd).dist_>cut
    rb.append((k,int(fe[ib].sum()),int(fm[ib].sum()),int(fm[~ib].sum())))
A['46B.3']=rb
fig,ax=small(2.4,1.4); ax.plot([r[0] for r in rb],[r[0] for r in rb],':',color=GREY); ax.plot([r[0] for r in rb],[r[1] for r in rb],'^-',color=RED,ms=4,label='empirical'); ax.plot([r[0] for r in rb],[r[2] for r in rb],'o-',color=GREEN,ms=3.5,label='MCD'); ax.set_xlabel('injected $k$'); ax.set_ylabel('caught'); ax.legend(loc='upper left',handlelength=1.2,fontsize=8.5); grid(ax); sv(fig,'p_46B3.png')
# 46C.1 variances against the target
S30=empirical_covariance(draw(30,0)); fig,ax=small(2.4,1.4); ax.bar(range(20),np.diag(S30),color=BLUE,width=0.7); ax.axhline(np.trace(S30)/20,color=RED,ls='--',lw=1.2); ax.set_xlabel('stock'); ax.set_ylabel('variance'); ax.set_xticks([0,9,19]); ax.set_xticklabels(['1','10','20']); sv(fig,'p_46C1.png')
A['46C.1']={'var_min':float(np.diag(S30).min()),'var_max':float(np.diag(S30).max()),'mu':float(np.trace(S30)/20)}
# 46C.2 max d2 of the empirical covariance
fig,ax=small(2.4,1.4); nn=np.arange(5,200); ax.plot(nn,nn-1,color=BLUE,lw=1.4); ax.axhline(cut,color=RED,ls='--',lw=1.2); ax.set_xlabel('$n$'); ax.set_ylabel('largest $d^2$'); ax.set_yscale('log'); grid(ax); sv(fig,'p_46C2.png')
A['46C.2']={'max_d2_n5':4,'max_d2_n71':70}
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1); print(CHK); print(json.dumps(A)[:1500])

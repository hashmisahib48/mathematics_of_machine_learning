# Problem figures of Chapter 43. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.decomposition import NMF, FastICA, LatentDirichletAllocation as LDA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score as ari
from corpus import make_docs
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.5): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 43A.1  one KL update of H
X=np.array([[3,6],[6,3.]]); W=np.array([[1,2],[2,1.]]); H=np.array([[1,1],[1,1.]])
kl=lambda X,Y: float(np.sum(X*np.log(X/Y)-X+Y))
Y0=W@H; H1=H*(W.T@(X/Y0))/W.sum(0)[:,None]; Y1=W@H1
A['43A.1']={'H1':H1.round(4).tolist(),'WH0':Y0.tolist(),'WH1':Y1.round(4).tolist(),'D0':round(kl(X,Y0),4),'D1':round(kl(X,Y1),4)}
fig,axs=plt.subplots(1,3,figsize=(2.2,1.0),dpi=240)
for ax,(M,t) in zip(axs,((X,'$\\mathbf{X}$'),(Y0,'$\\mathbf{WH}_0$'),(Y1,'$\\mathbf{WH}_1$'))):
    ax.imshow(M,cmap='Blues',vmin=0,vmax=9); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=9.5,color=NAVY,pad=2)
    for (i,j),v in np.ndenumerate(M): ax.text(j,i,f'{v:.2f}'.rstrip('0').rstrip('.') if abs(v-round(v))<1e-9 else f'{v:.2f}',ha='center',va='center',fontsize=9,color=NAVY)
    for sp_ in ax.spines.values(): sp_.set_visible(False)
sv(fig,'p_43A1.png')

# 43A.2  scale of the two losses
d2=lambda x,y:0.5*(x-y)**2; d1=lambda x,y:x*np.log(x/y)-x+y
A['43A.2']={'d2_4_2':d2(4,2),'d2_40_20':d2(40,20),'d1_4_2':round(float(d1(4,2)),4),'d1_40_20':round(float(d1(40,20)),4),'ratio2':d2(40,20)/d2(4,2),'ratio1':round(float(d1(40,20)/d1(4,2)),3)}
fig,ax=small(2.4,1.45); xs=np.arange(2); ax.bar(xs-0.2,[d2(4,2),d1(4,2)],0.38,color=BLUE,label='$x=4,\\ y=2$'); ax.bar(xs+0.2,[d2(40,20),d1(40,20)],0.38,color=OR,label='$x=40,\\ y=20$')
ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels(['$d_2$','$d_1$'],fontsize=9.5); ax.set_ylabel('divergence'); ax.set_ylim(0.3,900); grid(ax); ax.legend(loc='upper right',fontsize=8.5,handlelength=1.0,borderaxespad=0.1,labelspacing=0.2)
sv(fig,'p_43A2.png')

# 43A.3  kurtosis of three samples
smp={'a':np.array([-1,-1,-1,1,1,1.]),'b':np.array([-2,-1,0,0,1,2.]),'c':np.array([0,0,0,0,0,0,0,0,-3,3.])}
ek=lambda y:(float(np.mean((y-y.mean())**4)/np.mean((y-y.mean())**2)**2-3))
A['43A.3']={k:{'var':float(np.mean(v**2)),'m4':float(np.mean(v**4)),'kurt':round(ek(v),4)} for k,v in smp.items()}
fig,axs=plt.subplots(1,3,figsize=(2.2,1.0),dpi=240,sharey=False)
for ax,(k,v) in zip(axs,smp.items()):
    vals,cnt=np.unique(v,return_counts=True)
    for a_,c_ in zip(vals,cnt):
        for r_ in range(c_): ax.scatter(a_,r_+0.5,s=14,color=BLUE,zorder=3)
    ax.axhline(0,color=GREY,lw=0.8); ax.set_xlim(-3.8,3.8); ax.set_ylim(0,8.2); ax.set_yticks([]); ax.set_xticks([-3,0,3]); ax.set_title(f'sample {k}',fontsize=9.5,color=NAVY,pad=2); ax.spines['left'].set_visible(False)
sv(fig,'p_43A3.png')

# 43A.4  posterior mean of the topic mix
cnt=np.array([6,3,1.]); res={}
for a_ in (0.1,1,10): res[str(a_)]=((cnt+a_)/(cnt.sum()+3*a_)).round(4).tolist()
A['43A.4']=res
fig,ax=small(2.4,1.4); w=0.26
for k,(a_,col) in enumerate(((0.1,BLUE),(1,GREEN),(10,PUR))): ax.bar(np.arange(3)+(k-1)*w,res[str(a_)],w,color=col,label=f'$\\alpha$ = {a_:g}')
ax.set_xticks(range(3)); ax.set_xticklabels(['A','B','C']); ax.set_ylabel('$E[\\theta_t]$'); ax.set_ylim(0,0.95); grid(ax); ax.legend(loc='upper right',fontsize=8.5,handlelength=1.0,borderaxespad=0.1,labelspacing=0.2)
sv(fig,'p_43A4.png')

# data for B
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
# 43B.1  NMF test error and sparsity against k
rows=[]
for k in (5,10,15,20):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',ConvergenceWarning); m=NMF(k,max_iter=1000,random_state=0).fit(Xtr)
    Hk=m.components_; rows.append((k,round(float(np.linalg.norm(Xte-m.transform(Xte)@Hk)),1),round(float((Hk==0).mean()),3)))
A['43B.1']=rows
fig,ax=small(2.4,1.5); ax.plot([r[0] for r in rows],[r[1] for r in rows],'-o',color=GREEN,ms=3.8,lw=1.4); ax.set_xticks([5,10,15,20]); ax.set_xlabel('$k$'); ax.set_ylabel('test error'); ax.set_ylim(0,None); grid(ax)
sv(fig,'p_43B1.png')

# 43B.2  FastICA against noise
t=np.linspace(0,8,2000); S=np.c_[np.sin(2*t),2*(0.7*t%1)-1]; S=(S-S.mean(0))/S.std(0); Am=np.array([[1.0,1.0],[0.5,2.0]])
rng=np.random.RandomState(0); N0=rng.standard_normal(S.shape); sig=[0.05,0.2,0.5,1.0,2.0]; wm=[];wp=[]
from sklearn.decomposition import PCA
def worst(E): return float(np.abs(np.corrcoef(E.T,S.T)[:2,2:]).max(1).min())
for s_ in sig:
    Xm=S@Am.T+s_*N0; wm.append(round(worst(FastICA(2,random_state=0).fit_transform(Xm)),4)); wp.append(round(worst(PCA(2).fit_transform(Xm)),4))
A['43B.2']={'sigma':sig,'ica':wm,'pca':wp}
fig,ax=small(2.4,1.5); ax.plot(sig,wm,'-o',color=GREEN,ms=3.6,lw=1.4,label='FastICA'); ax.plot(sig,wp,'--s',color=BLUE,ms=3.4,lw=1.2,label='PCA'); ax.set_xlabel('noise level $\\sigma$'); ax.set_ylabel('worst $|$corr$|$'); ax.set_ylim(0.2,1.05); ax.set_xlim(-0.1,2.1); grid(ax); ax.legend(loc='lower left',fontsize=8.5,handlelength=1.4,borderaxespad=0.1,labelspacing=0.2)
sv(fig,'p_43B2.png')

# 43B.3  LDA priors
docs,hid=make_docs(0); Xc=CountVectorizer(stop_words='english',min_df=2).fit_transform(docs); pure=hid>=0; res=[]
for a_ in (0.01,0.1,1.0):
    l=LDA(4,doc_topic_prior=a_,max_iter=100,random_state=0).fit(Xc); th=l.transform(Xc)
    res.append((a_,round(float(th.max(1).mean()),3),round(float(ari(hid[pure],th[pure].argmax(1))),3)))
A['43B.3']=res
fig,ax=small(2.4,1.5); ax.bar(range(3),[r[1] for r in res],0.55,color=PUR); ax.set_xticks(range(3)); ax.set_xticklabels(['0.01','0.1','1']); ax.set_xlabel('doc_topic_prior'); ax.set_ylabel('mean largest $\\theta$'); ax.set_ylim(0.4,1.0); grid(ax)
sv(fig,'p_43B3.png')

# 43C.1  two factorisations of the paints
fig,ax=small(2.4,1.6); Xp=np.array([[4,2],[3,3],[2,4.]])
ax.fill([0,6,6,0],[0,0,6,6],color='#F3F4F6',zorder=0); ax.fill([0,5.4,2.7],[0,2.7,5.4],color='#E3F4EA',zorder=1)
ax.scatter(Xp[:,0],Xp[:,1],s=26,color=BLUE,edgecolor='white',lw=0.6,zorder=5)
for h in ([2,1],[1,2]): arr(ax,(0,0),(h[0]*1.25,h[1]*1.25),c=GREEN,lw=1.6,ms=7)
arr(ax,(0,0),(4.9,0),c=OR,lw=1.2,ms=7); arr(ax,(0,0),(0,4.9),c=OR,lw=1.2,ms=7)
ax.set_xlim(-0.2,5.5); ax.set_ylim(-0.2,5.2); ax.set_aspect('equal'); ax.set_xlabel('sensor 1'); ax.set_ylabel('sensor 2'); sv(fig,'p_43C1.png')
A['43C.1']={'identity_ok':bool(np.allclose(Xp@np.eye(2),Xp))}

# 43C.2  Gaussian sources
rng=np.random.RandomState(0); G=rng.standard_normal((250,2)); th=np.radians(35); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); Gr=G@R.T
fig,axs=plt.subplots(1,2,figsize=(2.9,1.35),dpi=240)
for ax,(D_,t_) in zip(axs,((G,'sources'),(Gr,'rotated'))):
    ax.scatter(D_[:,0],D_[:,1],s=3,color=BLUE,lw=0,alpha=0.7); ax.set_xlim(-3.6,3.6); ax.set_ylim(-3.6,3.6); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t_,fontsize=9.5,color=NAVY,pad=2)
    for sp_ in ax.spines.values(): sp_.set_visible(True); sp_.set_color(GRID)
A['43C.2']={'cov_after':np.cov(Gr.T).round(2).tolist()}
sv(fig,'p_43C2.png')

# 43C.3  perplexity of held-out documents
tr_,te_=slice(0,45),slice(45,60); Ks=list(range(1,9)); ptr=[];pte=[]
for K_ in Ks:
    a=[];b=[]
    for sd in (0,1,2):
        l=LDA(K_,max_iter=100,random_state=sd).fit(Xc[tr_]); a.append(l.perplexity(Xc[tr_])); b.append(l.perplexity(Xc[te_]))
    ptr.append(float(np.mean(a))); pte.append(float(np.mean(b)))
A['43C.3']={'K':Ks,'train':[round(v,1) for v in ptr],'test':[round(v,1) for v in pte]}
fig,ax=small(2.4,1.5); ax.plot(Ks,ptr,'-o',color=BLUE,ms=3.2,lw=1.3,label='training'); ax.plot(Ks,pte,'--s',color=RED,ms=3.2,lw=1.3,label='held out'); ax.set_xlabel('topics $K$'); ax.set_ylabel('perplexity'); grid(ax); ax.legend(loc='upper left',fontsize=8.5,handlelength=1.4,borderaxespad=0.1,labelspacing=0.2)
sv(fig,'p_43C3.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

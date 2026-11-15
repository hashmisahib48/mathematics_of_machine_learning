# Problem figures of Chapter 42. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split
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

# 42A.1 four points
P=np.array([[6,2],[3,2],[5,3],[6,5]],float); mu=P.mean(0); C=P-mu; S=C.T@C/3; ev,V=np.linalg.eigh(S); ev=ev[::-1]; V=V[:,::-1]
w1=V[:,0]*np.sign(V[:,0].sum()); z=C@w1
A['42A.1']={'mean':mu.tolist(),'S':S.tolist(),'eig':ev.tolist(),'w1':r4(w1),'scores':r4(z),'ratio':round(float(ev[0]/ev.sum()),4)}
fig,ax=small(2.4,1.7); ax.scatter(P[:,0],P[:,1],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.scatter(*mu,marker='+',s=60,color=NAVY,zorder=6)
for t in (-3,3): pass
ax.plot(mu[0]+np.array([-2.2,2.2])*w1[0],mu[1]+np.array([-2.2,2.2])*w1[1],color=GREEN,lw=1.3)
for k,(a,b) in enumerate(P): ax.text(a+0.15,b+0.15,str(k+1),fontsize=9,color=NAVY)
ax.set_xlim(1.5,8.5); ax.set_ylim(0.5,6); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal'); grid(ax); sv(fig,'p_42A1.png')

# 42A.2 choosing k
lam=np.array([4.2,2.4,1.6,1.1,0.7]); rr=lam/lam.sum(); cs=np.cumsum(rr)
kf={f:int(np.searchsorted(cs,f,side='right')+1) for f in (0.8,0.9,0.95)}
A['42A.2']={'ratio':r4(rr),'cum':r4(cs),'k':kf}
fig,ax=small(2.4,1.5); ax.bar(range(1,6),rr,color=BLUE,width=0.6); ax.plot(range(1,6),cs,'-o',color=GREEN,ms=3.5,lw=1.2); ax.set_xticks(range(1,6)); ax.set_xlabel('component $j$'); ax.set_ylabel('share'); ax.set_ylim(0,1.08); grid(ax); sv(fig,'p_42A2.png')

# 42A.3 singular values
sg=np.array([10,6,3,1.]); n=101; lm=sg**2/(n-1); k=2
A['42A.3']={'lam':r4(lm),'err_total':float((sg[k:]**2).sum()),'err_check':float((n-1)*lm[k:].sum()),'per_sample':round(float((sg[k:]**2).sum()/n),4),'kept':round(float(lm[:k].sum()/lm.sum()),4)}
fig,ax=small(2.4,1.4); ax.bar(range(1,5),sg,color=[BLUE,BLUE,GREY,GREY],width=0.6)
for i_,v in enumerate(sg): ax.text(i_+1,v+0.4,f'{v:g}',ha='center',fontsize=9,color=NAVY)
ax.set_xticks(range(1,5)); ax.set_xlabel('$j$'); ax.set_ylabel('$\\sigma_j$'); ax.set_ylim(0,12.5); grid(ax); sv(fig,'p_42A3.png')

# 42A.4 a new fish
FX=np.array([[27,4],[29,6],[30,6],[31,8],[33,6]],float); fm=FX.mean(0); fp=PCA(2).fit(FX); w1f=fp.components_[0]; G=np.array([28.,6.])
zg=fp.transform(G[None])[0]; back=fm+zg[0]*w1f; err=float(((G-back)**2).sum())
A['42A.4']={'scores':r4(zg),'back':r4(back),'err2':round(err,4),'z2_sq':round(float(zg[1]**2),4)}
fig,ax=small(2.4,1.7); ax.scatter(FX[:,0],FX[:,1],s=26,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.scatter(*G,marker='*',s=90,color=OR,edgecolor='white',lw=0.6,zorder=6)
ax.plot(fm[0]+np.array([-4,4])*w1f[0],fm[1]+np.array([-4,4])*w1f[1],color=GREEN,lw=1.2); ax.text(G[0]-0.2,G[1]+0.35,'G',fontsize=9,color=NAVY,ha='right')
ax.set_xlim(25,35); ax.set_ylim(3,9.5); ax.set_xlabel('length'); ax.set_ylabel('weight'); grid(ax); sv(fig,'p_42A4.png')

# data for B and C
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
# 42B.1 eigh
Xc=Xtr-Xtr.mean(0); Sx=Xc.T@Xc/(len(Xtr)-1); e,Vv=np.linalg.eigh(Sx); Vv=Vv[:,::-1][:,:5]; Zh=Xc@Vv
pc=PCA(5,svd_solver='full').fit(Xtr); Zp=pc.transform(Xtr)
sgn=np.sign((Zh*Zp).sum(0)); A['42B.1']={'flipped':[int(s<0) for s in sgn],'max_abs_diff_after_sign':float(np.abs(Zh*sgn-Zp).max()),'max_abs_diff_raw':round(float(np.abs(Zh-Zp).max()),3),'eig_agree':float(np.abs(e[::-1][:5]-pc.explained_variance_).max())}
A['42B.1']['max_abs_diff_after_sign']=float(f"{A['42B.1']['max_abs_diff_after_sign']:.2e}")
fig,ax=small(2.4,1.6); ax.scatter(Zp[:,1],Zh[:,1],s=3,color=PUR,alpha=0.6); ax.plot([-30,30],[-30,30],color=GREY,lw=0.8,ls=':'); ax.plot([-30,30],[30,-30],color=GREY,lw=0.8,ls=':')
ax.set_xlabel('PCA score 2'); ax.set_ylabel('eigh score 2'); grid(ax); sv(fig,'p_42B1.png')

# 42B.2 incremental batch sizes
full=PCA(10,svd_solver='full').fit(Xtr); bs=[50,100,200,400,800,1347]; rows=[]
for b in bs:
    ip=IncrementalPCA(10,batch_size=b).fit(Xtr); cos=np.abs((ip.components_*full.components_).sum(1)); rows.append((b,round(float(cos.min()),4),float(np.abs(ip.explained_variance_/full.explained_variance_-1).max())))
A['42B.2']=[(b,c,float(f'{v:.1e}')) for b,c,v in rows]
fig,ax=small(2.4,1.5); ax.plot([r[0] for r in rows],[r[1] for r in rows],'-o',color=PUR,ms=3.5,lw=1.3); ax.set_xscale('log'); ax.set_xticks([50,100,200,400,800,1347]); ax.set_xticklabels(['50','100','200','400','800','n'],fontsize=8.5)
ax.set_xlabel('batch size'); ax.set_ylabel('min $|\\cos|$'); grid(ax); sv(fig,'p_42B2.png')

# 42C.1 whitening of the smallest eigenvalues
sm=np.sort(np.linalg.eigvalsh(Sx))[:8]; A['42C.1']={'smallest_eig':[float(f'{v:.2e}') for v in sm],'constant_pixels':int((Xtr.std(0)==0).sum())}
fig,ax=small(2.4,1.5); ax.bar(range(1,9),np.maximum(sm,1e-17),color=[RED if v<1e-10 else GREY for v in sm],width=0.6); ax.set_yscale('log'); ax.set_ylim(1e-17,1e0)
ax.set_xlabel('the 8 smallest eigenvalues'); ax.set_ylabel('$\\lambda$'); ax.set_xticks([]); grid(ax); sv(fig,'p_42C1.png')

# 42C.2 wine variances
WX,WY=load_wine(return_X_y=True); vv=WX.var(0,ddof=1); A['42C.2']={'ratio_pc1':round(float(PCA(1).fit(WX).explained_variance_ratio_[0]),4),'max_var':round(float(vv.max()),2),'min_var':round(float(vv.min()),4)}
fig,ax=small(2.4,1.5); ax.bar(range(13),np.sort(vv)[::-1],color=BLUE,width=0.7); ax.set_yscale('log'); ax.set_xlabel('13 features, by variance'); ax.set_ylabel('variance'); ax.set_xticks([]); grid(ax); sv(fig,'p_42C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

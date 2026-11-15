# Problem figures of Chapter 2. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.datasets import load_iris, load_digits
A={}
def small(w=3.0,h=2.0): return plt.subplots(figsize=(w,h),dpi=240)
def arw(ax,p,q,c=BLUE,lw=1.6,ms=7,ls='-'): ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=ms,color=c,lw=lw,shrinkA=0,shrinkB=0,ls=ls,zorder=4))
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)

# 2A.1 norms
x=np.array([2.,-3,6]); z=np.array([1.,-3,2]); n2,n1,ni=[float(np.linalg.norm(x,q)) for q in (2,1,np.inf)]
A['2A.1']={'L2':n2,'L1':n1,'Linf':ni,'unit':np.round(x/n2,3).tolist(),'dist':round(float(np.linalg.norm(x-z)),3),'dist_sq':float(((x-z)**2).sum())}
fig,ax=small(2.8,1.6); ax.barh([2,1,0],[n2,n1,ni],color=[BLUE,OR,GREEN],edgecolor='white',hatch=None,height=0.62)
for yy,v_,l in zip([2,1,0],[n2,n1,ni],['$\\|x\\|_2$','$\\|x\\|_1$','$\\|x\\|_\\infty$']): ax.text(v_+0.2,yy,f'{l} = {v_:.0f}',va='center',fontsize=8.5,color=NAVY)
ax.set_xlim(0,17); ax.set_yticks([]); ax.set_xlabel('value for $\\mathbf{x}=(2,-3,6)$',fontsize=8.5); ax.tick_params(labelsize=7.5); ax.spines['left'].set_visible(False); save(fig,'p_2A1.png')

# 2A.2 projection
a=np.array([1.,2]); bb=np.array([4.,3]); c=float(a@bb/(a@a)); p=c*a; r=bb-p
A['2A.2']={'c':c,'p':p.tolist(),'r':r.tolist(),'a.r':float(a@r)}
fig,ax=small(2.3,1.6); ax.set_aspect('equal'); grid(ax); xx=np.array([-0.3,4.6]); ax.plot(xx,2*xx,color=GREEN,lw=0.8,ls=':')
arw(ax,(0,0),tuple(bb),OR,1.8); arw(ax,(0,0),tuple(p),GREEN,2.6); arw(ax,(0,0),tuple(a),BLUE,1.2); arw(ax,tuple(p),tuple(bb),RED,1.4)
ax.text(4.1,3.05,'$\\mathbf{b}$',color=OR,fontsize=9); ax.text(0.3,1.6,'$\\mathbf{a}$',color=BLUE,fontsize=9,ha='right'); ax.text(1.5,3.6,'$\\mathbf{p}$',color=GREEN,fontsize=9,ha='right'); ax.text(3.2,3.95,'$\\mathbf{r}$',color=RED,fontsize=9)
ax.set_xlim(-0.4,4.8); ax.set_ylim(-0.4,5.0); save(fig,'p_2A2.png')

# 2A.3 inverse
M=np.array([[3.,1],[5,2]]); Mi=np.linalg.inv(M); bv=np.array([1.,2]); xs=np.linalg.solve(M,bv)
A['2A.3']={'det':round(float(np.linalg.det(M)),6),'inv':np.round(Mi,6).tolist(),'x':np.round(xs,6).tolist()}
fig,ax=small(2.6,2.1); ax.set_aspect('equal'); grid(ax); Q=M@np.array([[0,0],[1,0],[1,1],[0,1]]).T; ax.add_patch(Polygon(Q.T,fc=OR,alpha=0.35,ec=OR,lw=1)); ax.add_patch(Polygon([[0,0],[1,0],[1,1],[0,1]],fc=BLUE,alpha=0.35,ec=BLUE,lw=1))
arw(ax,(0,0),(3,5),BLUE,1.6); arw(ax,(0,0),(1,2),GREEN,1.6); ax.text(3.1,5.0,'(3, 5)',fontsize=8,color=BLUE); ax.text(1.25,1.55,'(1, 2)',fontsize=8,color=GREEN); ax.text(0.1,-0.9,'unit square (blue) and its image',fontsize=7.8,color=GREY)
ax.set_xlim(-0.3,4.9); ax.set_ylim(-1.3,5.6); save(fig,'p_2A3.png')

# 2A.4 eigen
S=np.array([[3.,1],[1,3]]); lam,V=np.linalg.eigh(S); lam=lam[::-1]; V=V[:,::-1]
A['2A.4']={'lam':lam.tolist(),'v':np.round(V.T,4).tolist(),'trace':float(np.trace(S)),'det':float(np.linalg.det(S))}
fig,ax=small(2.4,1.8); ax.set_aspect('equal'); grid(ax); th=np.linspace(0,2*np.pi,200); Cc=np.c_[np.cos(th),np.sin(th)]; ax.plot(Cc[:,0],Cc[:,1],color=BLUE,lw=1.2); E=Cc@S.T; ax.plot(E[:,0],E[:,1],color=NAVY,lw=1.3)
for k,cc in enumerate([GREEN,PUR]):
    d=lam[k]*V[:,k]; ax.plot([-d[0],d[0]],[-d[1],d[1]],color=cc,lw=0.8,ls=':'); arw(ax,(0,0),tuple(d),cc,1.8)
ax.text(3.0,3.0,'$\\lambda_1\\mathbf{v}_1$',fontsize=9,color=GREEN); ax.text(1.5,-2.4,'$\\lambda_2\\mathbf{v}_2$',fontsize=9,color=PUR); ax.set_xlim(-4.4,4.4); ax.set_ylim(-3.6,4.1); save(fig,'p_2A4.png')

# 2B.1 iris class means, cosines
Xi,yi=load_iris(return_X_y=True); mu=np.array([Xi[yi==k].mean(0) for k in range(3)]); nm=mu/np.linalg.norm(mu,axis=1)[:,None]; Cm=nm@nm.T
A['2B.1']={'means':np.round(mu,3).tolist(),'cos_01':round(float(Cm[0,1]),4),'cos_02':round(float(Cm[0,2]),4),'cos_12':round(float(Cm[1,2]),4),'angle_01_deg':round(float(np.degrees(np.arccos(Cm[0,1]))),2),'angle_12_deg':round(float(np.degrees(np.arccos(Cm[1,2]))),2)}
fig,ax=small(2.3,0.78); im=ax.imshow(Cm,cmap='Blues',vmin=0.9,vmax=1.0,aspect='auto'); ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['setosa','versicolor','virginica'],fontsize=8); ax.set_yticklabels(['setosa','versicolor','virginica'],fontsize=8)
for i in range(3):
    for j in range(3): ax.text(j,i,f'{Cm[i,j]:.3f}',ha='center',va='center',fontsize=8,color=NAVY if Cm[i,j]<0.99 else 'white')
ax.set_title('cosine of the class means',fontsize=8.5,color=NAVY); save(fig,'p_2B1.png')

# 2B.2 rank of delivery X with miles
xd=np.array([1.,2,3,4,5]); Xm=np.column_stack([np.ones(5),xd,0.621371*xd]); sv=np.linalg.svd(Xm,compute_uv=False)
A['2B.2']={'rank':int(np.linalg.matrix_rank(Xm)),'sigma':[float(f'{v_:.4g}') for v_ in sv],'tol':float(sv.max()*max(Xm.shape)*np.finfo(float).eps)}
assert sv[2]<1e-12
fig,ax=small(2.8,1.45); ax.bar([1,2,3],np.maximum(sv,1e-18),color=[BLUE,BLUE,RED],width=0.55); ax.set_yscale('log'); ax.set_ylim(1e-17,50)
for k,v_ in enumerate(sv): ax.text(k+1,max(v_,1e-18)*2.2,f'{v_:.2g}',ha='center',fontsize=8,color=NAVY)
ax.set_xticks([1,2,3]); ax.set_xlabel('index $j$',fontsize=8.5); ax.set_ylabel('$\\sigma_j$',fontsize=9); ax.tick_params(labelsize=7.5); ax.grid(axis='y',color=GRID,lw=0.5); save(fig,'p_2B2.png')

# 2B.3 digits low rank
D=load_digits(); im0=D.images[0]; U,s,Vt=np.linalg.svd(im0); NA=np.linalg.norm(im0)
rel=[float(np.linalg.norm(im0-(U[:,:k]*s[:k])@Vt[:k])/NA) for k in range(1,5)]
A['2B.3']={'sigma':[round(float(v_),3) for v_ in s],'rel_err':[round(v_,4) for v_ in rel],'rank':int(np.linalg.matrix_rank(im0))}
fig,axs=plt.subplots(1,3,figsize=(3.4,1.35),dpi=240)
for ax,(k,t) in zip(axs,[(None,'original'),(1,'rank 1'),(3,'rank 3')]):
    ax.axis('off'); ax.imshow(im0 if k is None else (U[:,:k]*s[:k])@Vt[:k],cmap='gray_r',vmin=0,vmax=16); ax.set_title(t,fontsize=8,color=NAVY)
fig.tight_layout(w_pad=0.3); save(fig,'p_2B3.png')

# 2C.1 rank staircase
fig,ax=small(2.8,1.25); ps=np.arange(0,161); ax.plot(ps,np.minimum(100,ps),color=BLUE,lw=1.6); ax.axvline(120,color=RED,lw=0.9,ls='--'); ax.plot(120,100,'o',color=RED,ms=4)
ax.text(118,60,'$p=120$',color=RED,fontsize=8.5,ha='right'); ax.text(10,106,'$n=100$ samples: rank $\\leq 100$',fontsize=8,color=NAVY); ax.set_xlabel('number of columns $p$',fontsize=8.5); ax.set_ylabel('largest possible rank',fontsize=8.5)
ax.set_ylim(0,125); ax.set_xlim(0,160); ax.tick_params(labelsize=7.5); ax.grid(color=GRID,lw=0.5); save(fig,'p_2C1.png')

# 2C.2 cond of X and X^T X
rho=np.linspace(0,0.999,300); cx=np.sqrt((1+rho)/(1-rho)); fig,ax=small(2.9,1.6); ax.semilogy(rho,cx,color=BLUE,lw=1.6,label='cond($\\mathbf{X}$)'); ax.semilogy(rho,cx**2,color=RED,lw=1.6,ls='--',label='cond($\\mathbf{X}^{\\mathrm{T}}\\mathbf{X}$)')
ax.set_xlabel('correlation $\\rho$',fontsize=8.5); ax.set_ylabel('condition number',fontsize=8.5); ax.legend(fontsize=8,loc='upper left'); ax.tick_params(labelsize=7.5); ax.grid(color=GRID,lw=0.5); ax.set_xlim(0,1); save(fig,'p_2C2.png')

# 2C.3 documents
d1=np.array([3.,1]); d2=np.array([6.,2]); d3=np.array([1.,3]); cs=lambda a_,b_: float(a_@b_/np.linalg.norm(a_)/np.linalg.norm(b_))
A['2C.3']={'dist12':round(float(np.linalg.norm(d1-d2)),3),'dist13':round(float(np.linalg.norm(d1-d3)),3),'cos12':round(cs(d1,d2),4),'cos13':round(cs(d1,d3),4)}
fig,ax=small(2.6,2.0); ax.set_aspect('equal'); grid(ax)
for d_,c_,lw_ in [(d2,OR,1.3),(d3,GREEN,1.7),(d1,BLUE,2.6)]: arw(ax,(0,0),tuple(d_),c_,lw_)

ax.text(3.2,0.4,'$d_1=(3,1)$',color=BLUE,fontsize=8); ax.text(5.2,2.3,'$d_2=(6,2)$',color=OR,fontsize=8); ax.text(0.2,3.2,'$d_3=(1,3)$',color=GREEN,fontsize=8)
ax.set_xlabel('count of word 1',fontsize=8.5); ax.set_ylabel('count of word 2',fontsize=8.5); ax.set_xlim(-0.3,7.5); ax.set_ylim(-0.3,4.2); save(fig,'p_2C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A)[:1500])

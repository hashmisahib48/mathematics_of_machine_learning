# All figures of Chapter 46. Run: python3 fig/figs.py  (PNGs into ch46/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import chi2
from sklearn.datasets import load_wine
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance, MinCovDet, GraphicalLasso,
                                empirical_covariance, log_likelihood)
from sklearn.preprocessing import StandardScaler
from returns import true_model, draw
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def ellipse(ax,loc,cov,c2,**kw):
    lam,V=np.linalg.eigh(cov); t=np.linspace(0,2*np.pi,200); P=np.sqrt(c2)*(V*np.sqrt(lam))@np.vstack([np.cos(t),np.sin(t)])
    ax.plot(loc[0]+P[0],loc[1]+P[1],**kw)
Sigma,Theta,sd=true_model(); p=20
# ---- wine data with twelve injected samples (as in code/mcd_wine.py)
wine=load_wine(); yw=wine.target; cols=[0,6,9,12]; Xw=wine.data[:,cols]
bad=np.random.RandomState(0).choice(np.where(yw==2)[0],12,replace=False)
Wd=np.vstack([Xw[yw==0],Xw[bad]]); isbad=np.r_[np.zeros(59,bool),np.ones(12,bool)]
mcd=MinCovDet(random_state=0).fit(Wd); emp=EmpiricalCovariance().fit(Wd); cut=chi2(4).isf(0.025)

# ======== 46.0-1 opener
fig,axs=plt.subplots(1,2,figsize=(5.0,2.4),dpi=240,gridspec_kw={'width_ratios':[1,1.05]})
X30=draw(30,0); ax=axs[0]; idx=np.arange(1,21)
ev_t=np.sort(np.linalg.eigvalsh(Sigma))[::-1]; ev_s=np.sort(np.linalg.eigvalsh(empirical_covariance(X30)))[::-1]; ev_l=np.sort(np.linalg.eigvalsh(LedoitWolf().fit(X30).covariance_))[::-1]
ax.plot(idx,ev_t,'--',color=NAVY,lw=1.4,label='true'); ax.plot(idx,ev_s,'o-',color=BLUE,ms=3.5,lw=1,label='sample, $n=30$'); ax.plot(idx,ev_l,'s-',color=GREEN,ms=3.2,lw=1,label='Ledoit–Wolf')
ax.set_yscale('log'); ax.set_xlabel('eigenvalue number'); ax.set_ylabel('eigenvalue'); ax.set_xticks([1,5,10,15,20]); ax.legend(loc='lower left',handlelength=1.6,labelspacing=0.2); grid(ax); ax.set_title('20 stocks, 30 days',fontsize=9,color=NAVY)
ax=axs[1]; a,b=1,2   # flavanoids, colour intensity
ax.scatter(Wd[~isbad,a],Wd[~isbad,b],s=9,color=BLUE,lw=0,alpha=0.8,label='cultivar 1')
ax.scatter(Wd[isbad,a],Wd[isbad,b],s=16,marker='^',color=OR,lw=0,label='injected')
ix=[a,b]
ellipse(ax,emp.location_[ix],emp.covariance_[np.ix_(ix,ix)],chi2(2).isf(0.025),color=RED,lw=1.4,ls='--',label='empirical')
ellipse(ax,mcd.location_[ix],mcd.covariance_[np.ix_(ix,ix)],chi2(2).isf(0.025),color=GREEN,lw=1.6,label='MinCovDet')
ax.set_ylim(1,15.5); ax.set_xlabel('flavanoids'); ax.set_ylabel('colour intensity'); ax.set_title('wine with 12 impostors',fontsize=9,color=NAVY); ax.legend(loc='upper right',handlelength=1.4,labelspacing=0.1,fontsize=7.8,borderaxespad=0.1,ncol=2,columnspacing=0.8); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f460.png',470)
NUM['open']={'ev_true':[float(ev_t[0]),float(ev_t[-1])],'ev_sample':[float(ev_s[0]),float(ev_s[-1])]}

# ======== 46.2-1 intuition: Mahalanobis, correlation and partial correlation
fig,axs=plt.subplots(1,3,figsize=(5.0,1.95),dpi=240,gridspec_kw={'width_ratios':[1.25,1,1]})
T=np.array([[5,5],[6,9],[2,3],[4,1],[8,12.]]); m=T.mean(0); S=np.cov(T.T,bias=True); ax=axs[0]
for c2,ls in ((1,'-'),(4,'--')): ellipse(ax,m,S,c2,color=GREEN,lw=1.3,ls=ls)
ax.plot(T[:,0],T[:,1],'o',color=BLUE,ms=4.5); ax.plot(*m,'+',color=NAVY,ms=7)
for q,nm,dx,dy in (([9,12],'A',0.3,-0.2),([2,8],'B',-0.9,0.3)):
    ax.plot(*q,'s',color=RED,ms=5); ax.text(q[0]+dx,q[1]+dy,nm,color=RED,fontsize=9,weight='bold')
ax.plot([m[0],2],[m[1],8],color=RED,lw=0.8,ls=':'); ax.plot([m[0],9],[m[1],12],color=RED,lw=0.8,ls=':')
ax.set_xlim(-1.5,12); ax.set_ylim(-1,15); ax.set_xlabel('stock 1'); ax.set_ylabel('stock 2'); ax.set_title('contours $d^2=1$ and $4$',fontsize=8.5,color=NAVY); grid(ax)
C0=Sigma/np.outer(np.sqrt(np.diag(Sigma)),np.sqrt(np.diag(Sigma))); P0=-Theta/np.sqrt(np.outer(np.diag(Theta),np.diag(Theta))); np.fill_diagonal(P0,1)
for ax,M,tt in ((axs[1],C0,'correlation'),(axs[2],P0,'partial correlation')):
    im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1); ax.set_title(tt,fontsize=8.5,color=NAVY); ax.set_xticks([0,5,10,15,19]); ax.set_xticklabels(['1','6','11','16','20']); ax.set_yticks([0,5,10,15,19]); ax.set_yticklabels(['1','6','11','16','20']); ax.spines[:].set_visible(False)
    for s in (4.5,9.5,14.5): ax.axhline(s,color=GREY,lw=0.4); ax.axvline(s,color=GREY,lw=0.4)
NUM['heat']={'nonzero_corr_pairs':int((np.abs(C0)>1e-9).sum()-20)//2,'nonzero_partial_pairs':int((np.abs(P0)>1e-9).sum()-20)//2,'corr_adjacent':float(C0[2,3]),'partial_adjacent':float(P0[2,3]),'corr_min':float(C0[~np.eye(20,dtype=bool)].min())}
fig.tight_layout(w_pad=0.6); sv(fig,'f461.png',470)

# ======== 46.4-1 shrinkage: eigenvalue paths and risk
bv=json.load(open('verify/prose_numbers.json'))['bv']
fig,axs=plt.subplots(1,2,figsize=(5.0,2.3),dpi=240)
ax=axs[0]; S30=empirical_covariance(X30); mu=np.trace(S30)/p; als=np.linspace(0,1,50)
for l in np.linalg.eigvalsh(S30): ax.plot(als,(1-als)*l+als*mu,color=BLUE,lw=0.9)
ax.axhline(mu,color=GREY,lw=0.8,ls=':'); ax.text(0.80,mu*1.12,r'$\hat\mu$',fontsize=9,color=GREY)
ax.plot(np.zeros(20),ev_t[::-1],'_',color=NAVY,ms=7,mew=1.5); ax.set_yscale('log'); ax.set_xlabel(r'shrinkage $\alpha$'); ax.set_ylabel('eigenvalue of $\\hat{\\Sigma}_\\alpha$'); ax.set_title('eigenvalues move to $\\hat\\mu$',fontsize=9,color=NAVY); grid(ax)
ax.plot([],[],'_',color=NAVY,mew=1.5,label='true'); ax.legend(loc='lower right',handlelength=1.0,borderaxespad=0.2)
ax=axs[1]; al=np.array(bv['alphas']); ax.plot(al,bv['mse'],color=NAVY,lw=1.8,label='risk'); ax.plot(al,bv['bias2'],color=RED,lw=1.3,ls='--',label='bias$^2$'); ax.plot(al,bv['var'],color=BLUE,lw=1.3,ls='-.',label='variance')
ax.axvline(bv['best'],color=GREY,lw=0.8,ls=':'); ax.plot(bv['lw'],np.interp(bv['lw'],al,bv['mse']),'o',color=GREEN,ms=5,zorder=5,label='Ledoit–Wolf')
ax.plot(bv['oas'],np.interp(bv['oas'],al,bv['mse']),'s',color=OR,ms=4.5,zorder=5,label='OAS')
ax.set_xlabel(r'shrinkage $\alpha$'); ax.set_ylabel('squared error'); ax.set_title('trade-off, $n=30$',fontsize=9,color=NAVY); ax.legend(loc='upper center',handlelength=1.5,labelspacing=0.15,fontsize=7.8,ncol=2,columnspacing=0.8); ax.set_ylim(0,19.5); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f462.png',470)

# ======== 46.6-1 C-steps
rng=np.random.RandomState(11); good=rng.multivariate_normal([0,0],[[1,0.8],[0.8,1.2]],28); out=rng.multivariate_normal([3.5,-2.5],0.25*np.eye(2),12); XX=np.vstack([good,out]); n_=40; h=int(np.ceil((n_+3)/2))
def cs(H):
    mm=XX[H].mean(0); SS=np.cov(XX[H].T,bias=True); d2=np.einsum('ij,jk,ik->i',XX-mm,np.linalg.inv(SS),XX-mm); return np.argsort(d2)[:h],mm,SS
chosen=None
for seed in range(200):
    H0=np.random.RandomState(seed).permutation(n_)[:h]
    if (H0>=28).sum()<6: continue
    chain=[(H0,)+(XX[H0].mean(0),np.cov(XX[H0].T,bias=True))]; H=H0
    for it in range(8):
        Hn,mm,SS=cs(H); chain.append((Hn,XX[Hn].mean(0),np.cov(XX[Hn].T,bias=True)))
        if set(Hn)==set(H): break
        H=Hn
    if len(chain)>=4 and (chain[-1][0]>=28).sum()==0: chosen=(seed,chain); break
seed,chain=chosen; steps=[0,1,2,len(chain)-2]; chain_dets=[float(np.linalg.det(c[2])) for c in chain]
fig,axs=plt.subplots(1,4,figsize=(5.0,1.7),dpi=240,sharex=True,sharey=True)
for ax,k in zip(axs,steps):
    H,mm,SS=chain[k]; msk=np.zeros(n_,bool); msk[H]=True
    ax.plot(XX[~msk,0],XX[~msk,1],'o',color=GRID,mec=GREY,mew=0.5,ms=3.2); ax.plot(XX[msk,0],XX[msk,1],'o',color=GREEN,ms=3.2)
    ellipse(ax,mm,SS,chi2(2).isf(0.025),color=NAVY,lw=1.1); ax.set_title(('start' if k==0 else f'step {k}')+f'\n$\\det={np.linalg.det(SS):.2f}$',fontsize=8.5,color=NAVY); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(w_pad=0.3); sv(fig,'f463.png',470)
NUM['cstep']={'seed':int(seed),'dets':chain_dets,'n_outliers_in_start':int((chain[0][0]>=28).sum()),'h':h,'steps_shown':steps}

# ======== 46.8-1 flow chart
fig,ax=plt.subplots(figsize=(5.0,2.55),dpi=240); ax.set_xlim(0,10); ax.set_ylim(0,6.6); ax.axis('off')
def row(y,name,items):
    ax.text(0.0,y+0.95,name,fontsize=8.5,color=RED,weight='bold',va='center'); x=0.0; gap=0.3
    for k_,(w_,txt,kind) in enumerate(items):
        fc,ec={'d':(LB,BLUE),'s':(LGREY,GREY),'o':('#E3F4EA',GREEN),'m':(PINK,RED)}[kind]
        box(ax,x,y-0.5,w_,1.0,txt,fc=fc,ec=ec,fs=7.8); x+=w_
        if k_<len(items)-1: arr(ax,(x,y),(x+gap,y),c=NAVY,lw=1,ms=7); x+=gap
row(5.0,'shrinkage',[(0.9,'data\n$\\mathbf{X}$','d'),(1.6,'$\\mathbf{S}$, $\\hat\\mu$\nEq. 46.3-3','s'),(2.2,'$\\alpha$ fixed, LW\nor OAS','s'),(2.3,'$(1-\\alpha)\\mathbf{S}+\\alpha\\hat\\mu\\mathbf{I}$\nEq. 46.4-1','o')])
row(2.9,'graphical lasso',[(0.9,'data\n$\\mathbf{X}$','d'),(1.0,'$\\mathbf{S}$','s'),(2.5,'for each row $j$:\nlasso, Eq. 46.5-4','s'),(1.4,'gap $<$ tol?\nEq. 46.5-5','m'),(1.9,'sparse $\\hat{\\boldsymbol{\\Theta}}$\nand $\\hat{\\boldsymbol{\\Sigma}}$','o')])
row(0.8,'MinCovDet',[(0.9,'data\n$\\mathbf{X}$','d'),(1.9,'30 random starts,\n2 C-steps each','s'),(1.6,'best 10 to\nconvergence','s'),(1.4,'scale by $c$,\nreweight','s'),(1.9,'location, $\\hat{\\boldsymbol{\\Sigma}}$,\nsupport, dist','o')])
sv(fig,'f464.png',430)

# ======== 46.11-1 hyperparameters: shrinkage, graphical lasso CV, support_fraction
PN=json.load(open('verify/prose_numbers.json')); gl=PN['glcv']
Xte=draw(2000,99); fig,axs=plt.subplots(1,3,figsize=(5.0,2.15),dpi=240)
ax=axs[0]; als=np.linspace(0.02,1,50); lls=[ShrunkCovariance(shrinkage=float(a)).fit(X30).score(Xte) for a in als]
ax.plot(als,lls,color=PUR,lw=1.5); lw_=LedoitWolf().fit(X30); oa_=OAS().fit(X30)
ax.plot(lw_.shrinkage_,lw_.score(Xte),'o',color=GREEN,ms=5,label='LW'); ax.plot(oa_.shrinkage_,oa_.score(Xte),'s',color=OR,ms=4.5,label='OAS')
ax.axhline(log_likelihood(empirical_covariance(Xte,assume_centered=True),np.linalg.inv(Sigma)),color=GREY,ls=':',lw=1); ax.text(0.45,-36.3,'true $\\Sigma$',fontsize=8,color=GREY)
ax.set_ylim(-46,-35.5); ax.set_xlabel(r'$\alpha$ (shrinkage)'); ax.set_ylabel('test log-likelihood'); ax.legend(loc='lower right',handlelength=1.0,fontsize=8,labelspacing=0.15); grid(ax); ax.set_title('shrinkage',fontsize=8.5,color=NAVY)
NUM['shr']={'lw':float(lw_.shrinkage_),'lw_ll':float(lw_.score(Xte)),'oas_ll':float(oa_.score(Xte)),'best_alpha':float(als[int(np.argmax(lls))]),'best_ll':float(max(lls)),'ll_alpha_0.02':float(lls[0]),'ll_true':float(log_likelihood(empirical_covariance(Xte,assume_centered=True),np.linalg.inv(Sigma)))}
ax=axs[1]; a_=np.array(gl['alphas'][1:]); mn=np.array(gl['mean'][1:]); sd_=np.array(gl['std'][1:])
ax.plot(a_,mn,'o-',color=PUR,ms=2.8,lw=1.2); ax.fill_between(a_,mn-sd_,mn+sd_,color=PUR,alpha=0.15,lw=0); ax.axvline(gl['best'],color=GREEN,lw=1.2); ax.axhline(gl['mean'][0],color=RED,ls='--',lw=1.1)
ax.text(0.2,gl['mean'][0]+0.3,r'$\alpha=0$',fontsize=8,color=RED); ax.set_xscale('log'); ax.set_xlabel(r'$\alpha$ (penalty)'); ax.set_ylabel('CV log-likelihood'); ax.set_ylim(-31,-23); ax.set_title('graphical lasso',fontsize=8.5,color=NAVY); grid(ax)
ax=axs[2]; sf=np.array(PN['sf']); ax.plot(sf[:,0],sf[:,1],'o-',color=PUR,ms=3.2,lw=1.3); ax.axhline(np.linalg.norm(emp.covariance_-EmpiricalCovariance().fit(Wd[~isbad]).covariance_)/np.linalg.norm(EmpiricalCovariance().fit(Wd[~isbad]).covariance_),color=RED,ls='--',lw=1.1)
ax.axvline(59/71,color=GREY,ls=':',lw=1); ax.text(0.80,0.30,'clean\nfraction',fontsize=7.8,color=GREY,ha='right'); ax.plot(38/71,0.116,'o',color=GREEN,ms=6,mfc='none',mew=1.4); ax.text(0.52,0.54,'empirical',fontsize=8,color=RED)
ax.set_xlabel('`support_fraction`'.replace('`','')); ax.set_ylabel('relative error'); ax.set_ylim(0,0.75); ax.set_title('MinCovDet',fontsize=8.5,color=NAVY); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f465.png',470)

# ======== 46.11-2 graphs recovered at three penalties (n = 150)
Z=StandardScaler().fit_transform(draw(150,1)); edge=lambda P:(np.abs(P)>1e-10)&~np.eye(len(P),dtype=bool); tr_=edge(Theta)
pos={i:(i%5,3-i//5) for i in range(p)}
fig,axs=plt.subplots(1,4,figsize=(5.0,1.55),dpi=240); rec={}
for ax,(a_,tt) in zip(axs,((None,'true graph'),(0.1,r'$\alpha=0.1$'),(0.2,r'$\alpha=0.2$'),(0.4,r'$\alpha=0.4$'))):
    E_=tr_ if a_ is None else edge(GraphicalLasso(alpha=a_,enet_tol=1e-6).fit(Z).precision_)
    for i in range(p):
        for j in range(i+1,p):
            if E_[i,j] or tr_[i,j]:
                col,ls=(GREEN,'-') if (E_[i,j] and tr_[i,j]) else ((RED,'--') if E_[i,j] else (BLUE,':'))
                ax.plot([pos[i][0],pos[j][0]],[pos[i][1],pos[j][1]],color=col,ls=ls,lw=1.0 if col!=GREEN else 1.3)
    ax.plot([v[0] for v in pos.values()],[v[1] for v in pos.values()],'o',color=NAVY,ms=2.6,zorder=5); ax.set_xlim(-0.5,4.5); ax.set_ylim(-0.5,3.5); ax.axis('off')
    k=int(E_.sum()//2); tp=int((E_&tr_).sum()//2); ax.set_title(tt+f'\n{k} edges, {tp} true',fontsize=8,color=NAVY); rec[str(a_)]=(k,tp)
fig.tight_layout(w_pad=0.2); sv(fig,'f466.png',470); NUM['graphs']=rec

# ======== 46.12-1 case study 1: n against error
CR=json.load(open('verify/case_returns.json')); fig,axs=plt.subplots(1,2,figsize=(5.0,2.3),dpi=240)
sty={'Empirical':(RED,'-^'),'LedoitWolf':(GREEN,'-o'),'OAS':(OR,'--s'),'GraphLassoCV':(BLUE,'-D')}
X7=draw(2000,777); llt=log_likelihood(empirical_covariance(X7,assume_centered=True),np.linalg.inv(Sigma)); NUM['case_llt']=float(llt)

for k,(c_,mk) in sty.items():
    v=np.array(CR[k]); axs[0].plot(v[:,0],v[:,1],mk,color=c_,ms=3.5,lw=1.2,label={'LedoitWolf':'Ledoit–Wolf','GraphLassoCV':'graphical lasso'}.get(k,k))
    axs[1].plot(v[:,0],np.maximum(llt-v[:,2],1e-3)+0.0,mk,color=c_,ms=3.5,lw=1.2)
axs[0].axvline(20,color=GREY,ls=':',lw=1); axs[0].text(21,0.5,'$p$',fontsize=8.5,color=GREY)
for ax in axs: ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('days $n$'); grid(ax)
axs[0].set_ylabel('squared error'); axs[1].set_ylabel('log-likelihood shortfall'); axs[0].legend(loc='upper right',handlelength=1.6,labelspacing=0.15,fontsize=8)
for ax in axs: ax.set_xticks([30,60,120,240,480]); ax.set_xticklabels(['30','60','120','240','480']); ax.minorticks_off()
fig.tight_layout(w_pad=1.0); sv(fig,'f467.png',470)

# ======== 46.12-2 case study 2: distances on the wine data
fig,axs=plt.subplots(1,2,figsize=(5.0,2.2),dpi=240)
for ax,d2,tt in ((axs[0],emp.mahalanobis(Wd),'empirical'),(axs[1],mcd.dist_,'MinCovDet')):
    ii=np.arange(len(d2)); ax.plot(ii[~isbad],d2[~isbad],'o',color=BLUE,ms=3,label='cultivar 1'); ax.plot(ii[isbad],d2[isbad],'^',color=OR,ms=4.2,label='injected')
    ax.axhline(cut,color=RED,lw=1.2,ls='--'); ax.set_yscale('log'); ax.set_xlabel('sample'); ax.set_title(tt,fontsize=9,color=NAVY); grid(ax); ax.set_ylim(0.05,3000)
axs[0].set_ylabel('squared Mahalanobis distance'); axs[0].legend(loc='upper left',handlelength=1.0,labelspacing=0.15,fontsize=8,borderaxespad=0.1,ncol=1)
axs[1].text(2,cut*2.2,'cut-off 11.14',fontsize=8,color=RED)
axs[0].text(30,cut*2.2,'cut-off',fontsize=8,color=RED)
fig.tight_layout(w_pad=1.0); sv(fig,'f468.png',470)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})

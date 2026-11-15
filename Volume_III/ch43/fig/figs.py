# All figures of Chapter 43. Run: python3 fig/figs.py [part ...]  (PNGs into ch43/, numbers into verify/fig_numbers.json)
import sys,os,json,io,runpy,warnings,contextlib; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Ellipse, Polygon, Rectangle
from sklearn.decomposition import PCA, NMF, FastICA, FactorAnalysis, MiniBatchDictionaryLearning
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
ARGS=sys.argv[1:]; want=lambda k: (not ARGS) or (k in ARGS)
NUMF='verify/fig_numbers.json'; NUM=json.load(open(NUMF)) if (ARGS and os.path.exists(NUMF)) else {}; CHK=json.load(open('verify/fig_textsize.json')) if (ARGS and os.path.exists('verify/fig_textsize.json')) else {}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def quiet(path):
    with contextlib.redirect_stdout(io.StringIO()): return runpy.run_path(path)
DX,DY=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(DX,DY,test_size=0.25,random_state=0,stratify=DY)
LG='#E3F4EA'

# ---------------------------------------------------------------- 43.0-1 the frame
if want('f430'):
    fig,ax=plt.subplots(figsize=(5.4,2.55),dpi=240); ax.set_xlim(0,10.8); ax.set_ylim(0,5.2); ax.axis('off')
    def blk(x,y,w,h,fc,ec,lab,sub=None):
        ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=1.1,zorder=2)); ax.text(x+w/2,y+h/2+(0.15 if sub else 0),lab,ha='center',va='center',fontsize=11,color=NAVY,weight='bold',zorder=3)
        if sub: ax.text(x+w/2,y+h/2-0.38,sub,ha='center',va='center',fontsize=8.5,color=NAVY,zorder=3)
    blk(0.25,1.1,1.5,2.7,LB,BLUE,'X','data'); ax.text(0.25+0.75,0.85,'n × p',ha='center',va='top',fontsize=8.5,color=GREY)
    ax.text(2.18,2.45,'≈',fontsize=18,ha='center',va='center',color=NAVY)
    blk(2.6,1.1,0.9,2.7,LG,GREEN,'W','codes'); ax.text(2.6+0.45,0.85,'n × k',ha='center',va='top',fontsize=8.5,color=GREY)
    blk(3.7,2.2,1.5,1.6,LG,GREEN,'H','parts'); ax.text(3.7+0.75,1.95,'k × p',ha='center',va='top',fontsize=8.5,color=GREY)
    ax.text(0.25,4.65,'rows: samples, columns: features',fontsize=8.5,color=GREY,va='center'); ax.text(0.25,4.3,'row t of H is part t (components_)',fontsize=8.5,color=GREY,va='center')
    rows=[('PCA (Chapter 42)','rows of H orthonormal; W = scores',GREY),('NMF','W ≥ 0, H ≥ 0',BLUE),('FastICA','columns of W independent, non-Gaussian',BLUE),('FactorAnalysis','W Gaussian; one noise variance per feature',BLUE),('DictionaryLearning','W sparse; rows of H of length ≤ 1',BLUE),('LatentDirichletAllocation','rows of W and H are probabilities',BLUE)]
    ax.plot([5.55,5.55],[0.25,4.95],color=GRID,lw=0.8)
    for k,(a,b_,col) in enumerate(rows):
        y=4.55-0.78*k
        ax.text(5.8,y+0.12,a,fontsize=8.8,color=col,weight='bold',va='center'); ax.text(5.8,y-0.22,b_,fontsize=8.5,color=NAVY if col!=GREY else GREY,va='center')
    fig.subplots_adjust(0,0,1,1); sv(fig,'f430.png',470)

# ---------------------------------------------------------------- 43.1-1 the three paints
if want('f431'):
    Xp=np.array([[4,2],[3,3],[2,4]],float); Hs=np.array([[2,1],[1,2]],float)
    fig,ax=plt.subplots(figsize=(3.5,2.7),dpi=240)
    ax.add_patch(Polygon([[0,0],[5.2,2.6],[2.6,5.2]],fc='#E3F4EA',ec='none',zorder=1))
    for h,nm in zip(Hs,('pigment 1','pigment 2')):
        arr(ax,(0,0),tuple(h*1.15),c=GREEN,lw=2.0,ms=9); 
    ax.text(2.45,1.15,'part 1 = (2, 1)',color=GREEN,fontsize=8.5,ha='left',va='top'); ax.text(0.45,2.75,'part 2 = (1, 2)',color=GREEN,fontsize=8.5,ha='left',va='bottom')
    ax.plot([0,5.2],[0,0],color=GRID,lw=0.8,ls='--',zorder=1); ax.plot([0,0],[0,5.2],color=GRID,lw=0.8,ls='--',zorder=1)
    ax.scatter(Xp[:,0],Xp[:,1],s=48,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
    for k,(a,b_) in enumerate(Xp): ax.text(a+0.14,b_+0.14,f'paint {k+1}',fontsize=8.5,color=NAVY)
    ax.set_xlim(-0.2,5.2); ax.set_ylim(-0.2,5.0); ax.set_aspect('equal'); ax.set_xlabel('sensor 1 reading'); ax.set_ylabel('sensor 2 reading'); grid(ax)
    fig.tight_layout(); sv(fig,'f431.png',340)

# ---------------------------------------------------------------- 43.2-1 geometry
if want('f432'):
    rng=np.random.RandomState(1); Ht=np.array([[1.0,0.25],[0.3,1.0]]); S_=rng.exponential(1.0,(300,2)); Xg=S_@Ht+0.02*rng.standard_normal((300,2))
    pca=PCA(2).fit(Xg); nmf=NMF(2,init='nndsvda',max_iter=2000,random_state=0).fit(Xg); ica=FastICA(2,random_state=0,whiten='unit-variance').fit(Xg)
    dl=MiniBatchDictionaryLearning(3,alpha=0.5,batch_size=50,max_iter=50,fit_algorithm='lars',transform_algorithm='omp',transform_n_nonzero_coefs=1,random_state=0).fit(Xg-Xg.mean(0))
    Hn=nmf.components_; M=ica.mixing_.T*1.0
    ang=lambda v:float(np.degrees(np.arctan2(v[1],v[0])))
    NUM['geom']={'true_angles':[ang(h) for h in Ht],'pca_angles':[ang(v) for v in pca.components_],'nmf_angles':[ang(v) for v in Hn],'ica_angles':[ang(v) for v in M],'dict_angles':[ang(v) for v in dl.components_],
                 'pca_dot':float(pca.components_[0]@pca.components_[1]),'ica_cos':float((M[0]@M[1])/np.linalg.norm(M[0])/np.linalg.norm(M[1])),'nmf_cos':float(Hn[0]@Hn[1]/np.linalg.norm(Hn[0])/np.linalg.norm(Hn[1]))}
    fig,axs=plt.subplots(2,2,figsize=(4.9,4.2),dpi=240)
    mu=Xg.mean(0)
    for ax,ttl in zip(axs.ravel(),('PCA: orthogonal axes through the mean','NMF: rays from the origin','FastICA: independent axes through the mean','DictionaryLearning: 3 atoms')):
        ax.scatter(Xg[:,0],Xg[:,1],s=6,color=BLUE,lw=0,alpha=0.55,zorder=2); ax.set_xlim(-0.4,5.0); ax.set_ylim(-0.4,4.2); ax.set_aspect('equal'); ax.set_title(ttl,fontsize=8.8,color=NAVY); grid(ax)
    ax=axs[0,0]
    for v,l in zip(pca.components_,pca.explained_variance_): arr(ax,tuple(mu),tuple(mu+2.0*np.sqrt(l)*v),c=GREEN,lw=2.0,ms=9)
    ax.scatter(*mu,marker='+',s=40,color=NAVY,lw=1.3,zorder=6)
    ax=axs[0,1]
    for v in Hn: arr(ax,(0,0),tuple(v*4.0/np.linalg.norm(v)),c=GREEN,lw=2.0,ms=9)
    ax=axs[1,0]
    for v in M:
        u=v/np.linalg.norm(v); sgn=np.sign(u.sum()); u=u*sgn; arr(ax,tuple(mu),tuple(mu+2.3*u),c=GREEN,lw=2.0,ms=9)
    ax.scatter(*mu,marker='+',s=40,color=NAVY,lw=1.3,zorder=6)
    ax=axs[1,1]
    for v in dl.components_:
        u=v/np.linalg.norm(v); u=u*np.sign(u.sum()); arr(ax,tuple(mu),tuple(mu+2.6*u),c=GREEN,lw=2.0,ms=9)
    ax.scatter(*mu,marker='+',s=40,color=NAVY,lw=1.3,zorder=6)
    for ax in axs[1]: ax.set_xlabel('feature 1')
    for ax in axs[:,0]: ax.set_ylabel('feature 2')
    fig.tight_layout(w_pad=0.6,h_pad=0.8); sv(fig,'f432.png',470)

# ---------------------------------------------------------------- 43.3-1 beta divergences and convergence
if want('f433'):
    fig,axs=plt.subplots(1,2,figsize=(5.4,2.35),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
    ax=axs[0]; y=np.linspace(0.12,3.0,400); x=1.0
    d2=0.5*(x-y)**2; d1=x*np.log(x/y)-x+y; d0=x/y-np.log(x/y)-1
    ax.plot(y,d2,color=BLUE,lw=1.6,label='β = 2 (Frobenius)'); ax.plot(y,d1,color=GREEN,lw=1.6,ls='--',label='β = 1 (KL)'); ax.plot(y,d0,color=RED,lw=1.4,ls=':',label='β = 0 (Itakura–Saito)')
    ax.axvline(1.0,color=GRID,lw=0.8); ax.set_ylim(0,1.6); ax.set_xlim(0,3.0); ax.set_xlabel('model value $y$ (the observation is $x=1$)'); ax.set_ylabel('$d_\\beta(1\\,|\\,y)$'); grid(ax)
    ax.legend(loc='upper right',fontsize=8.3,handlelength=1.8,borderaxespad=0.2)
    NUM['beta']={'d2_at_0.5':float(0.5*0.25),'d1_at_0.5':float(np.log(2)-1+0.5),'d0_at_0.5':float(2-np.log(2)-1),'d2_at_2':0.5,'d1_at_2':float(-np.log(2)+1),'d0_at_2':float(0.5-np.log(0.5)-1)}
    ax=axs[1]
    rng=np.random.RandomState(0); W0=rng.uniform(0.5,1.5,(len(Xtr),10)); H0=rng.uniform(0.5,1.5,(10,64))
    its=[1,2,3,5,8,12,20,30,50,80,120,200,300,500]; E_={'mu':[],'cd':[]}
    for sol in ('mu','cd'):
        for n_ in its:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',ConvergenceWarning)
                m=NMF(10,solver=sol,init='custom',max_iter=n_,tol=0.0).fit(Xtr,W=W0.copy(),H=H0.copy())
            E_[sol].append(float(m.reconstruction_err_))
    pe=PCA(10).fit(Xtr); pca_err=float(np.linalg.norm(Xtr-pe.inverse_transform(pe.transform(Xtr))))
    ax.plot(its,E_['mu'],'-o',color=BLUE,ms=3,lw=1.3,label="solver 'mu'"); ax.plot(its,E_['cd'],'--s',color=GREEN,ms=3,lw=1.3,label="solver 'cd'")
    ax.axhline(pca_err,color=GREY,lw=0.9,ls=':'); ax.text(500,pca_err+10,'PCA, 10 components',fontsize=8.3,color=GREY,ha='right',va='bottom')
    ax.set_xscale('log'); ax.set_xlabel('iterations (same start)'); ax.set_ylabel('$\\Vert\\mathbf{X}-\\mathbf{W}\\mathbf{H}\\Vert_F$'); ax.set_ylim(600,1500); grid(ax); ax.legend(loc='upper right',fontsize=8.3,handlelength=1.8,borderaxespad=0.2)
    NUM['conv']={'its':its,'mu':[round(v,2) for v in E_['mu']],'cd':[round(v,2) for v in E_['cd']],'pca':round(pca_err,2)}
    fig.tight_layout(w_pad=0.9); sv(fig,'f433.png',470)

# ---------------------------------------------------------------- 43.4-1 ICA on the toy
if want('f434'):
    T=quiet('code/toy_ica.py'); S_=T['S']; X_=T['X']; Z_=T['Z']; K_=T['K']
    fig,axs=plt.subplots(1,4,figsize=(5.4,1.85),dpi=240,gridspec_kw={'width_ratios':[1,1,1,1.25]})
    for ax,(D_,ttl,lim) in zip(axs[:3],((S_,'sources $\\mathbf{S}$',2.0),(X_,'mixtures $\\mathbf{X}$',3.8),(Z_,'whitened $\\mathbf{Z}$',1.9))):
        ax.axhline(0,color=GRID,lw=0.7); ax.axvline(0,color=GRID,lw=0.7)
        ax.scatter(D_[:,0],D_[:,1],s=30,color=BLUE,edgecolor='white',lw=0.7,zorder=5)
        ax.plot(*np.vstack([D_,D_[:1]]).T,color=GRID,lw=0.9,zorder=2)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect('equal'); ax.set_title(ttl,fontsize=8.8,color=NAVY); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values(): sp_.set_visible(False)
    th=np.radians(np.linspace(0,180,361)); m4=np.array([np.mean((Z_@[np.cos(t),np.sin(t)])**4) for t in th])
    ax=axs[3]; ax.plot(np.degrees(th),m4-3,color=GREEN,lw=1.6); ax.axhline(0,color=GREY,lw=0.8,ls=':'); ax.text(178,0.08,'Gaussian',fontsize=8.3,color=GREY,ha='right',va='bottom',zorder=6,bbox=dict(fc='white',ec='none',pad=0.6))
    w_true=np.array([0.8507,-0.5257]); a1=np.degrees(np.arctan2(w_true[1],w_true[0]))%180
    ax.axvline(a1,color=PUR,lw=1.0,ls='--'); ax.axvline((a1+90)%180,color=PUR,lw=1.0,ls='--')
    ax.set_xlim(0,180); ax.set_ylim(-2.15,0.5); ax.set_xticks([0,60,120,180]); ax.set_xlabel('angle of $\\mathbf{w}$ (degrees)'); ax.set_ylabel('kurt$(\\mathbf{w}^{T}\\mathbf{z})$'); grid(ax); ax.set_title('kurtosis',fontsize=8.8,color=NAVY)
    NUM['ica_toy']={'min_E_y4':float(m4.min()),'argmin_deg':float(np.degrees(th[m4.argmin()])),'max_E_y4':float(m4.max()),'a1':float(a1),'a2':float((a1+90)%180)}
    fig.tight_layout(w_pad=0.4); sv(fig,'f434.png',470)

# ---------------------------------------------------------------- 43.5-1 factor analysis against PCA
if want('f435'):
    F=quiet('code/fa_sklearn.py'); load=F['load']; psi=F['psi']; fa=F['fa']; pca=F['pca']
    lf=np.abs(fa.components_[0]); lp=np.abs(pca.components_[0])*np.sqrt(pca.explained_variance_[0])
    NUM['fa']={'load':load.tolist(),'fa_load':lf.round(3).tolist(),'pca_load':lp.round(3).tolist(),'psi':psi.tolist(),'fa_psi':fa.noise_variance_.round(3).tolist(),'pca_sigma2':float(pca.noise_variance_)}
    fig,axs=plt.subplots(1,2,figsize=(5.4,2.1),dpi=240,gridspec_kw={'width_ratios':[1.25,1]})
    xs=np.arange(6); w=0.26; ax=axs[0]
    ax.bar(xs-w,load,w,color=GRID,edgecolor=GREY,lw=0.5,label='true'); ax.bar(xs,lf,w,color=GREEN,label='FactorAnalysis'); ax.bar(xs+w,lp,w,color=OR,label='PCA')
    ax.set_xticks(xs); ax.set_xticklabels([f'x{j+1}' for j in xs]); ax.set_ylabel('loading'); ax.set_ylim(0,1.75); grid(ax); ax.legend(loc='upper center',ncol=3,fontsize=8,handlelength=1.0,columnspacing=0.8,borderaxespad=0.1,handletextpad=0.4)
    ax=axs[1]; ax.bar(xs-w/1.5,psi,w*1.3,color=GRID,edgecolor=GREY,lw=0.5,label='true'); ax.bar(xs+w/1.5,fa.noise_variance_,w*1.3,color=GREEN,label='FactorAnalysis')
    ax.axhline(pca.noise_variance_,color=OR,lw=1.6,ls='--'); ax.text(-0.4,pca.noise_variance_+0.1,'PCA: one value',fontsize=8.3,color=OR,ha='left',va='bottom')
    ax.set_xticks(xs); ax.set_xticklabels([f'x{j+1}' for j in xs]); ax.set_ylabel('noise variance $\\psi$'); ax.set_ylim(0,3.6); grid(ax)
    fig.tight_layout(w_pad=0.9); sv(fig,'f435.png',470)

# ---------------------------------------------------------------- 43.6-1 dictionary and denoising
if want('f436'):
    from sklearn.decomposition import SparseCoder
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d
    from scipy.ndimage import gaussian_filter
    D=quiet('code/dict_sklearn.py'); H=D['H']; img=D['img']; noisy=D['noisy']; Pc=D['Pc']; mu=D['mu']; rmse=D['rmse']
    coder=SparseCoder(H,transform_algorithm='omp',transform_n_nonzero_coefs=2); rec=(coder.transform(Pc)@H+mu).reshape(-1,8,8); den=reconstruct_from_patches_2d(rec,img.shape)
    blur=gaussian_filter(noisy,1.0)
    NUM['dict']={'rmse_noisy':float(rmse(noisy)),'rmse_dict2':float(rmse(den)),'rmse_blur':float(rmse(blur))}
    fig=plt.figure(figsize=(5.4,2.55),dpi=240)
    gl=fig.add_gridspec(4,8,wspace=0.06,hspace=0.06,left=0.01,right=0.55,top=0.90,bottom=0.04)
    gr=fig.add_gridspec(2,2,wspace=0.05,hspace=0.28,left=0.60,right=0.995,top=0.93,bottom=0.01)
    for k in range(32):
        a_=fig.add_subplot(gl[k//8,k%8]); v=H[k].reshape(8,8); m=np.abs(v).max(); a_.imshow(v,cmap='RdBu_r',vmin=-m,vmax=m,interpolation='nearest'); a_.set_xticks([]); a_.set_yticks([])
        for sp_ in a_.spines.values(): sp_.set_visible(False)
    fig.text(0.28,0.965,'32 learned atoms (rows of $\\mathbf{H}$)',fontsize=8.8,color=NAVY,ha='center',va='center')
    ims=[(img,'clean'),(noisy,'noisy: %.3f'%rmse(noisy)),(den,'2 atoms: %.3f'%rmse(den)),(blur,'blur: %.3f'%rmse(blur))]
    for k,(im,ttl) in enumerate(ims):
        a_=fig.add_subplot(gr[k//2,k%2]); a_.imshow(im,cmap='gray',vmin=-0.3,vmax=1.3,interpolation='nearest'); a_.set_xticks([]); a_.set_yticks([]); a_.set_title(ttl,fontsize=8.8,color=NAVY,pad=2)
        for sp_ in a_.spines.values(): sp_.set_visible(False)
    sv(fig,'f436.png',470)

# ---------------------------------------------------------------- 43.7-1 plate diagram and Dirichlet
if want('f437'):
    fig=plt.figure(figsize=(5.4,3.2),dpi=240)
    ax=fig.add_axes([0,0.43,1,0.57]); ax.set_xlim(0,10.8); ax.set_ylim(0.12,3.77); ax.axis('off')
    def node(x,y,t,sub,fill=False):
        ax.add_patch(Circle((x,y),0.4,fc=LGREY if fill else 'white',ec=NAVY,lw=1.2,zorder=3)); ax.text(x,y,t,ha='center',va='center',fontsize=11,color=NAVY,zorder=4)
        ax.text(x,y-0.62,sub,ha='center',va='top',fontsize=8.8,color=GREY)
    def link(a,b): arr(ax,a,b,c=NAVY,lw=1.0,ms=8)
    ya=2.1
    node(0.7,ya,'α','prior'); node(2.5,ya,'θ','topic mix'); node(4.3,ya,'z','topic'); node(6.1,ya,'w','word',fill=True); node(8.0,ya,'β','word probs.'); node(10.0,ya,'η','prior')
    link((1.12,ya),(2.08,ya)); link((2.92,ya),(3.88,ya)); link((4.72,ya),(5.68,ya)); link((7.58,ya),(6.52,ya)); link((9.58,ya),(8.42,ya))
    ax.add_patch(Rectangle((3.55,0.75),3.25,2.35,fc='none',ec=GREY,lw=1.0)); ax.text(6.65,0.82,'N',fontsize=9,color=GREY,ha='right',va='bottom',style='italic')
    ax.add_patch(Rectangle((1.75,0.3),5.25,3.1,fc='none',ec=GREY,lw=1.0,ls='--')); ax.text(6.85,0.36,'D',fontsize=9,color=GREY,ha='right',va='bottom',style='italic')
    ax.add_patch(Rectangle((7.3,0.75),1.4,2.35,fc='none',ec=GREY,lw=1.0)); ax.text(8.6,0.82,'K',fontsize=9,color=GREY,ha='right',va='bottom',style='italic')
    rng=np.random.RandomState(0)
    tri=np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2]])
    for k,a_ in enumerate((0.1,1.0,10.0)):
        axs=fig.add_axes([0.04+k*0.325,0.02,0.28,0.39]); th=rng.dirichlet([a_]*3,200); P=th@tri
        axs.add_patch(Polygon(tri,fc='none',ec=GREY,lw=0.9)); axs.scatter(P[:,0],P[:,1],s=6,color=PUR,lw=0,alpha=0.7)
        axs.set_xlim(-0.12,1.12); axs.set_ylim(-0.12,0.98); axs.set_aspect('equal'); axs.axis('off')
        axs.text(0.5,-0.08,f'α = {a_:g}',fontsize=8.8,color=NAVY,ha='center',va='top')
        for (x,y),l in zip(tri,('1','2','3')):
            axs.text(x+(-0.04 if x==0 else (0.04 if x==1 else 0)),y+(0.03 if y>0 else -0.03),l,fontsize=8.8,color=GREY,ha='right' if x==0 else ('left' if x==1 else 'center'),va='bottom' if y>0 else 'top')
    sv(fig,'f437.png',470)

if ARGS or True:
    pass
json.dump(NUM,open(NUMF,'w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print('figs ok',[k for k,v in CHK.items() if v['min_font_printed']<7.5])

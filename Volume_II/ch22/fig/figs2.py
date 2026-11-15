# Figures of Chapter 22, part 2: soft margin, kernels, galleries. Run: python3 fig/figs2.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image; w=Image.open(name).size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
OFF=LinearSegmentedColormap.from_list('bo',['#BBD3F2','#FFFFFF','#FBD9A8'])
def pts(ax,X,y,s=22,zorder=5):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.6,zorder=zorder)
def rings(ax,P,s=70,col=NAVY): ax.scatter(P[:,0],P[:,1],s=s,facecolor='none',edgecolor=col,lw=1.0,zorder=6)
def region(ax,m,lim,levels_margin=True,n=220):
    gx,gy=np.meshgrid(np.linspace(lim[0],lim[1],n),np.linspace(lim[2],lim[3],n)); F=m.decision_function(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx,gy,F,levels=[-1e3,0,1e3],colors=['#DCE8F8','#FBE7C9'],zorder=0)
    ax.contour(gx,gy,F,levels=[0],colors=GREEN,linewidths=1.8,zorder=2)
    if levels_margin: ax.contour(gx,gy,F,levels=[-1,1],colors=GREEN,linewidths=0.9,linestyles='--',zorder=2)
# ---------- soft-margin data
Xo,yo=make_classification(n_samples=40,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,class_sep=0.8,flip_y=0.0,random_state=10)
Xo=StandardScaler().fit_transform(Xo); so=2*yo-1
LIM=(-2.6,2.6,-2.6,2.6)
m=SVC(kernel='linear',C=1.0,tol=1e-8).fit(Xo,yo); a=np.abs(m.dual_coef_[0]); z=so*m.decision_function(Xo)
free=m.support_[a<1-1e-7]; bound=m.support_[a>=1-1e-7]; out=np.setdiff1d(np.arange(len(yo)),m.support_)
xi=np.maximum(0,1-z)
NUM['soft']={'n':len(yo),'C':1.0,'n_free':len(free),'n_bound':len(bound),'n_outside':len(out),'n_wrong':int((z<0).sum()),'inside_not_wrong':int(((z>=0)&(z<1-1e-6)).sum()),'margin':round(float(2/np.linalg.norm(m.coef_)),3),'sum_xi':round(float(xi.sum()),3),'train_acc':round(float(m.score(Xo,yo)),3)}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.9),dpi=240,gridspec_kw={'width_ratios':[1.15,1]}); ax=axs[0]
region(ax,m,LIM)
w=m.coef_[0]; nw=np.linalg.norm(w); u=w/nw
for i in bound:     # slack: distance from the point to its own margin line
    q=Xo[i]+(xi[i]/nw)*so[i]*u; ax.plot([Xo[i,0],q[0]],[Xo[i,1],q[1]],color=RED,lw=1.3,zorder=4)
pts(ax,Xo,yo); rings(ax,Xo[free]); rings(ax,Xo[bound],col=RED)
ax.set_xlim(*LIM[:2]); ax.set_ylim(*LIM[2:]); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_title('three kinds of points',fontsize=9.5,color=NAVY)
ax=axs[1]; zz=np.linspace(-1.6,2.4,200); ax.plot(zz,np.maximum(0,1-zz),color=RED,lw=1.8)
ax.axvline(1,color=GRID,lw=0.8); ax.axvline(0,color=GRID,lw=0.8)
for idx,mk,ec in ((out,'o',GREY),(free,'o',NAVY),(bound,'o',RED)):
    ax.scatter(z[idx],xi[idx],s=22 if idx is out else 30,facecolor='white' if idx is out else ('white' if idx is free else RED),edgecolor=ec,lw=1.0,zorder=5)
ax.set_xlabel('$y\\,f(\\mathbf{x})$'); ax.set_ylabel('slack  $\\xi=\\max(0,1-yf)$'); ax.set_title('hinge loss of each seed',fontsize=9.5,color=NAVY); ax.set_xlim(-1.6,2.4); ax.set_ylim(-0.12,2.7)
from matplotlib.lines import Line2D
H=[Line2D([],[],marker='o',ls='',mfc='white',mec=GREY,ms=5,label='$\\alpha=0$: outside the street'),Line2D([],[],marker='o',ls='',mfc='white',mec=NAVY,ms=7,mew=1.2,label='$0<\\alpha<C$: on the margin'),Line2D([],[],marker='o',ls='',mfc=RED,mec=RED,ms=6,label='$\\alpha=C$: inside or wrong side')]
fig.legend(handles=H,loc='lower center',ncol=3,fontsize=8.2,handletextpad=0.2,columnspacing=0.9,bbox_to_anchor=(0.5,-0.01))
fig.tight_layout(w_pad=1.0,rect=(0,0.09,1,1)); sv(fig,'f241.png',470)

# ---------- kernels: circles, linear against RBF
Xc,yc=make_circles(n_samples=120,noise=0.1,factor=0.45,random_state=1)
LC=(-1.7,1.7,-1.7,1.7); fig,axs=plt.subplots(1,2,figsize=(4.9,2.5),dpi=240)
NUM['circles']={}
for ax,(kern,ttl) in zip(axs,(('linear','linear kernel'),('rbf','RBF kernel, $\\gamma=1$'))):
    mm=SVC(kernel=kern,C=1.0,gamma=1.0).fit(Xc,yc); region(ax,mm,LC); pts(ax,Xc,yc,s=14); rings(ax,mm.support_vectors_,s=40)
    ax.set_xlim(*LC[:2]); ax.set_ylim(*LC[2:]); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_title(ttl+f': accuracy {mm.score(Xc,yc):.2f}',fontsize=9,color=NAVY)
    NUM['circles'][kern]={'acc':round(float(mm.score(Xc,yc)),3),'n_sv':int(len(mm.support_))}
axs[0].set_ylabel('$x_2$'); axs[1].set_yticklabels([])
fig.tight_layout(w_pad=0.8); sv(fig,'f251.png',470)
# ---------- C gallery on overlapping linear data
Xg,yg=make_classification(n_samples=60,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,class_sep=0.8,flip_y=0.0,random_state=10)
Xg=StandardScaler().fit_transform(Xg); sg=2*yg-1
fig,axs=plt.subplots(2,2,figsize=(4.9,4.55),dpi=240,sharex=True,sharey=True); NUM['Cgallery']={}
for ax,C in zip(axs.ravel(),(0.01,0.1,1,100)):
    mm=SVC(kernel='linear',C=C,tol=1e-8).fit(Xg,yg); region(ax,mm,LIM); pts(ax,Xg,yg,s=14); rings(ax,mm.support_vectors_,s=44)
    zz=sg*mm.decision_function(Xg); wn=np.linalg.norm(mm.coef_)
    NUM['Cgallery'][str(C)]={'n_sv':int(len(mm.support_)),'margin':round(float(2/wn),3),'norm_w':round(float(wn),3),'acc':round(float(mm.score(Xg,yg)),3),'n_margin_err':int((zz<1-1e-6).sum()),'n_wrong':int((zz<0).sum())}
    ax.set_xlim(*LIM[:2]); ax.set_ylim(*LIM[2:]); ax.set_aspect('equal'); ax.set_title(f'$C$ = {C:g}:  {len(mm.support_)} support vectors',fontsize=9,color=PUR,weight='bold')
for ax in axs[1]: ax.set_xlabel('$x_1$')
for ax in axs[:,0]: ax.set_ylabel('$x_2$')
fig.tight_layout(h_pad=0.6,w_pad=0.5); sv(fig,'f2111.png',470)
# ---------- gamma x C on make_moons
Xm,ym=make_moons(n_samples=300,noise=0.28,random_state=0); Xm_tr,Xm_te,ym_tr,ym_te=train_test_split(Xm,ym,test_size=0.5,random_state=0,stratify=ym)
LM=(-1.8,2.8,-1.3,1.8); fig,axs=plt.subplots(3,3,figsize=(4.9,4.45),dpi=240,sharex=True,sharey=True); NUM['moons']={}
gs=(0.1,1.0,10.0); Cs=(0.1,1.0,100.0)
for r_,g in enumerate(gs):
    for c_,C in enumerate(Cs):
        ax=axs[r_,c_]; mm=SVC(C=C,gamma=g).fit(Xm_tr,ym_tr); region(ax,mm,LM,levels_margin=False,n=160); pts(ax,Xm_tr,ym_tr,s=7); rings(ax,mm.support_vectors_,s=18)
        tr,te=mm.score(Xm_tr,ym_tr),mm.score(Xm_te,ym_te); NUM['moons'][f'g{g}_C{C}']={'n_sv':int(len(mm.support_)),'train':round(float(tr),3),'test':round(float(te),3)}
        ax.set_xlim(*LM[:2]); ax.set_ylim(*LM[2:]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{len(mm.support_)} SV;  test {te:.2f}',fontsize=8.5,color=NAVY,pad=2)
for r_,g in enumerate(gs): axs[r_,0].set_ylabel(f'$\\gamma$ = {g:g}',fontsize=9.5,color=PUR,weight='bold')
for c_,C in enumerate(Cs): axs[2,c_].set_xlabel(f'$C$ = {C:g}',fontsize=9.5,color=PUR,weight='bold')
fig.tight_layout(h_pad=0.5,w_pad=0.3); sv(fig,'f2112.png',470)
# ---------- validation heat map and curve
Cg=2.0**np.arange(-4,11,2); gg=2.0**np.arange(-8,7,2); cv=StratifiedKFold(5,shuffle=True,random_state=0)
H=np.array([[cross_val_score(SVC(C=C,gamma=g),Xm_tr,ym_tr,cv=cv).mean() for C in Cg] for g in gg])
bi=np.unravel_index(H.argmax(),H.shape); NUM['heat']={'best_gamma':float(gg[bi[0]]),'best_C':float(Cg[bi[1]]),'best_cv':round(float(H.max()),4),'min_cv':round(float(H.min()),4)}
mb=SVC(C=Cg[bi[1]],gamma=gg[bi[0]]).fit(Xm_tr,ym_tr); NUM['heat']['test_at_best']=round(float(mb.score(Xm_te,ym_te)),3)
fig,axs=plt.subplots(1,2,figsize=(4.9,2.55),dpi=240,gridspec_kw={'width_ratios':[1.1,1]}); ax=axs[0]
im=ax.imshow(H,origin='lower',cmap='Blues',vmin=0.7,vmax=0.95,aspect='auto')
for i_ in range(H.shape[0]):
    for j_ in range(H.shape[1]):
        ax.text(j_,i_,f'{100*H[i_,j_]:.0f}',ha='center',va='center',fontsize=6.6,color='white' if H[i_,j_]>0.85 else NAVY) if False else None
ax.scatter(bi[1],bi[0],marker='*',s=90,color=OR,edgecolor='white',lw=0.6,zorder=5)
ax.set_xticks(range(0,len(Cg),2)); ax.set_xticklabels([f'$2^{{{int(np.log2(v))}}}$' for v in Cg[::2]],fontsize=8); ax.set_yticks(range(0,len(gg),2)); ax.set_yticklabels([f'$2^{{{int(np.log2(v))}}}$' for v in gg[::2]],fontsize=8)
ax.set_xlabel('$C$'); ax.set_ylabel('$\\gamma$'); ax.set_title('5-fold accuracy (darker: better)',fontsize=9,color=NAVY)
ax=axs[1]; gb=gg[bi[0]]; trs=[];cvs=[]
for C in Cg:
    mm=SVC(C=C,gamma=gb).fit(Xm_tr,ym_tr); trs.append(mm.score(Xm_tr,ym_tr)); cvs.append(H[bi[0],list(Cg).index(C)])
ax.semilogx(Cg,trs,'-o',color=BLUE,ms=3,lw=1.3,label='training'); ax.semilogx(Cg,cvs,'--s',color=GREY,ms=3,lw=1.3,label='validation')
ax.axvline(Cg[bi[1]],color=PUR,lw=1); ax.set_xlabel('$C$'); ax.set_ylabel('accuracy'); ax.legend(loc='lower left',fontsize=8.5); ax.grid(color=GRID,lw=0.5)
ax.set_title(f'$\\gamma$ = {gb:g} fixed',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0); sv(fig,'f2113.png',470)
NUM['heat']['curve']={'C':[float(v) for v in Cg],'train':[round(float(v),3) for v in trs],'cv':[round(float(v),3) for v in cvs]}
json.dump(NUM,open('verify/fig_numbers2.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize2.json','w'),indent=1); print(json.dumps(NUM)); print(CHK)

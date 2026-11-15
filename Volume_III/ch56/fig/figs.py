# All figures of Chapter 56. Run: python3 fig/figs.py  (PNGs into ch56/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
NUM={}; CHK={}
J=lambda f: json.load(open('verify/'+f))
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
CM=LinearSegmentedColormap.from_list('sc',['#ECEFF4','#BCD3F2',BLUE,NAVY])
cs=J('case_search.json'); pool=J('pool.json')

# ======== 56.0-1 opener: the landscape, grid and random search
gsc=np.array(cs['grid_scores']).reshape(8,9)          # rows: C, columns: gamma
lc=np.arange(-5,10,2); lg=np.arange(-15,2,2)
fig,axs=plt.subplots(1,2,figsize=(4.9,1.95),dpi=240,gridspec_kw={'width_ratios':[1,1],'wspace':0.34})
ax=axs[0]; vmin,vmax=0.90,0.995
im=ax.imshow(gsc.T,origin='lower',cmap=CM,vmin=vmin,vmax=vmax,extent=(-6,10,-16,2),aspect='auto',interpolation='nearest')
bi=np.unravel_index(np.argmax(gsc),gsc.shape); NUM['grid_best_cell']=[int(lc[bi[0]]),int(lg[bi[1]])]
ties=np.argwhere(gsc>=gsc.max()-1e-12)
for t_ in ties: ax.plot(lc[t_[0]],lg[t_[1]],marker='*',ms=9,color=OR,mec=NAVY,mew=0.6,ls='')
ax.set_xlabel('log$_2\\,C$'); ax.set_ylabel('log$_2\\,\\gamma$'); ax.set_title('grid: 72 cells',fontsize=8,color=NAVY); ax.set_xticks([-4,0,4,8]); ax.set_yticks([-14,-10,-6,-2])
ax=axs[1]; rp=np.array(cs['random_points']); sc_=ax.scatter(np.log2(rp[:,0]),np.log2(rp[:,1]),c=rp[:,2],cmap=CM,vmin=vmin,vmax=vmax,s=22,edgecolor=NAVY,linewidth=0.4)
b=np.argmax(rp[:,2]); ax.plot(np.log2(rp[b,0]),np.log2(rp[b,1]),marker='*',ms=9,color=OR,mec=NAVY,mew=0.6,ls='')
ax.set_xlim(-6,10); ax.set_ylim(-16,2); ax.set_xlabel('log$_2\\,C$'); ax.set_title('random: 72 draws',fontsize=8,color=NAVY); ax.set_xticks([-4,0,4,8]); ax.set_yticks([-14,-10,-6,-2]); ax.set_yticklabels([])
cb=fig.colorbar(im,ax=axs,fraction=0.035,pad=0.02,extend='min'); cb.ax.tick_params(labelsize=7.8); cb.set_label('5-fold accuracy',fontsize=8)
sv(fig,'f560.png',470)

# ======== 56.2-1 grid against random, seen from the important parameter
fig,axs=plt.subplots(2,2,figsize=(4.4,2.2),dpi=240,gridspec_kw={'height_ratios':[0.7,1.25],'hspace':0.08,'wspace':0.18})
xs=np.linspace(0,1,200); fx=np.exp(-((xs-0.62)/0.13)**2)
rs_=np.random.RandomState(3); pts=rs_.rand(9,2)
gp=np.array([[a,b_] for a in (1/6,0.5,5/6) for b_ in (1/6,0.5,5/6)])
for k,(P_,ttl) in enumerate(((gp,'grid: 9 points'),(pts,'random: 9 points'))):
    a=axs[0,k]; a.plot(xs,fx,color=BLUE,lw=1.4); a.fill_between(xs,fx,color=LB); a.plot(P_[:,0],np.zeros(9)-0.06,'|',color=RED,ms=7,mew=1.4,clip_on=False)
    a.set_xlim(0,1); a.set_ylim(-0.12,1.2); a.set_xticks([]); a.set_yticks([]); a.spines['left'].set_visible(False); a.spines['bottom'].set_visible(False)
    a.set_title(ttl,fontsize=8,color=NAVY,pad=3)
    if k==0: a.text(0.0,1.12,'score',fontsize=7.8,color=NAVY,va='top')
    b_=axs[1,k]; b_.scatter(P_[:,0],P_[:,1],s=26,color=BLUE,edgecolor=NAVY,lw=0.5,zorder=3); b_.set_xlim(0,1); b_.set_ylim(0,1)
    b_.axvspan(0.5,0.74,color=LGREY,zorder=0); b_.set_xticks([]); b_.set_yticks([]); b_.set_xlabel('important parameter',fontsize=8); 
    if k==0: b_.set_ylabel('unimportant parameter',fontsize=8)
    nd=len(np.unique(np.round(P_[:,0],6))); NUM['distinct_values_%d'%k]=int(nd)
    b_.text(0.5,0.5,'',fontsize=7.8)
axs[1,0].text(0.97,0.04,'3 distinct values',ha='right',fontsize=7.8,color=RED); axs[1,1].text(0.97,0.04,'9 distinct values',ha='right',fontsize=7.8,color=RED)
sv(fig,'f562.png',400)

# ======== 56.3-1 nested cross-validation
fig,axs=plt.subplots(1,2,figsize=(4.8,1.75),dpi=240,gridspec_kw={'width_ratios':[1,1.15],'wspace':0.12})
ax=axs[0]; n=10
for k in range(5):
    for j in range(n): ax.add_patch(Rectangle((j,4-k),0.92,0.78,fc=OR if (j//2)==k else BLUE,ec='white',lw=0.4))
    ax.text(-0.35,4-k+0.39,str(k+1),ha='right',va='center',fontsize=7.8,color=NAVY)
ax.set_xlim(-1.2,10); ax.set_ylim(-0.2,5.4); ax.axis('off'); ax.set_title('outer: score the chosen model',fontsize=8,color=NAVY,pad=2)
ax.text(5,5.1,'',fontsize=7.8)
ax.add_patch(Rectangle((-0.1,3.95),10.22,0.98,fc='none',ec=RED,lw=1.2))
ax=axs[1]; m=8
for k in range(4):
    for j in range(m): ax.add_patch(Rectangle((j,3-k),0.92,0.78,fc=PUR if (j//2)==k else BLUE,ec='white',lw=0.4))
    ax.text(-0.35,3-k+0.39,str(k+1),ha='right',va='center',fontsize=7.8,color=NAVY)
ax.add_patch(Rectangle((m+0.2,0),1.7,3.78,fc=OR,ec='white'))
ax.text(m+1.05,1.9,'outer\ntest\nfold 1',ha='center',va='center',fontsize=7.8,color=NAVY,rotation=0)
ax.set_xlim(-1.2,10.4); ax.set_ylim(-0.2,5.4); ax.axis('off'); ax.set_title('inner: choose $\\theta$ without fold 1',fontsize=8,color=NAVY,pad=2)
ax.text(4,-0.05,'purple: inner validation',ha='center',va='top',fontsize=7.8,color=PUR); ax.set_ylim(-0.7,4.8)
sv(fig,'f563.png',470)

# ======== 56.5-1 the algorithm
fig,ax=plt.subplots(figsize=(4.9,1.8),dpi=240); ax.set_xlim(-0.1,13.0); ax.set_ylim(-0.3,6.3); ax.axis('off')
X4=[0.0,3.45,6.9,10.35]; WB=2.55
box(ax,X4[0],3.3,WB,1.2,'estimator,\ngrid or draws',fc=LB,ec=BLUE,fs=8.5)
box(ax,X4[1],3.3,WB,1.2,'candidates\n$\\theta_1,\\dots,\\theta_M$',fc='#EFE6F5',ec=PUR,fs=8.5)
box(ax,X4[2],3.3,WB,1.2,'clone, fit, score\nfor each $(\\theta_j,k)$',fc='#E3F2E9',ec=GREEN,fs=8.5)
box(ax,X4[3],3.3,WB,1.2,'mean, sd, rank\nper candidate',fc=LGREY,ec=GREY,fs=8.5)
box(ax,X4[3],0.5,WB,1.2,'best_index_\n= rank 1',fc=PINK,ec=RED,fs=8.5)
box(ax,X4[2],0.5,WB,1.2,'refit on\nall rows',fc='#E3F2E9',ec=GREEN,fs=8.5)
box(ax,X4[1],0.5,WB,1.2,'best_estimator_\nbest_params_',fc=LB,ec=BLUE,fs=8.5)
for p_,q_ in (((X4[0]+WB,3.9),(X4[1],3.9)),((X4[1]+WB,3.9),(X4[2],3.9)),((X4[2]+WB,3.9),(X4[3],3.9)),((X4[3]+WB/2,3.3),(X4[3]+WB/2,1.7)),((X4[3],1.1),(X4[2]+WB,1.1)),((X4[2],1.1),(X4[1]+WB,1.1))): arr(ax,p_,q_,c=NAVY,lw=1.0,ms=7)
ax.annotate('',xy=(X4[1]+WB/2,4.55),xytext=(X4[3]+WB/2,4.55),arrowprops=dict(arrowstyle='-|>',color=PUR,lw=1.0,ls='--',connectionstyle='arc3,rad=0.35',mutation_scale=7))
ax.text(7.2,5.95,'halving: keep the best $\\lceil n/\\eta\\rceil$, rows $\\times\\,\\eta$',ha='center',fontsize=8,color=PUR)
ax.text(0.0,1.1,'$K$ folds,\n$K\\,M+1$ fits',fontsize=8,color=NAVY,va='center')
sv(fig,'f565.png',440)

# ======== 56.8-1 behaviour: budget, optimism
cv=np.array(pool['cv']); te=np.array(pool['test']); M=len(cv); rng=np.random.RandomState(0); NP=1000
runcv=np.zeros((NP,M)); runte=np.zeros((NP,M))
for i in range(NP):
    p=rng.permutation(M); c_=cv[p]; best=-1; bt=0
    for j in range(M):
        if c_[j]>best: best=c_[j]; bt=te[p[j]]
        runcv[i,j]=best; runte[i,j]=bt
Ms=np.arange(1,M+1); q=lambda a,pc: np.percentile(a,pc,axis=0)
W=np.array(J('winner.json')); gr=pool['grids']; gk=sorted(int(k) for k in gr)
NUM['random_median_cv']={str(m):round(float(q(runcv,50)[m-1]),4) for m in (9,25,64,192)}
NUM['random_median_test']={str(m):round(float(q(runte,50)[m-1]),4) for m in (9,25,64,192)}
fig,axs=plt.subplots(1,3,figsize=(4.9,1.8),dpi=240,gridspec_kw={'width_ratios':[1.05,1.05,1.0],'wspace':0.62})
for k,(arr_,key,yl,ttl,yy) in enumerate(((runcv,0,'best 5-fold accuracy','(a) the search score',(0.90,1.0)),(runte,1,'test accuracy of the winner','(b) the same winners',(0.90,1.0)))):
    ax=axs[k]; ax.fill_between(Ms,q(arr_,10),q(arr_,90),color=LB,lw=0); ax.plot(Ms,q(arr_,50),color=BLUE,lw=1.3,label='random')
    ax.plot(gk,[gr[str(m)][key] for m in gk],'s',color=GREEN,ms=3.8,label='grid $m\\times m$',mec=NAVY,mew=0.4)
    ax.plot([3*m for m in gk],[gr[str(m)][key] for m in gk],'s',mfc='white',mec=GREEN,ms=3.8,mew=0.9,label='grid, inert 3rd')
    ax.set_xscale('log'); ax.set_xlim(3,300); ax.set_ylim(*yy); ax.set_xlabel('candidates'); ax.set_ylabel(yl,fontsize=7.8); ax.set_title(ttl,fontsize=8,color=NAVY); grid(ax)
    if k==0: ax.legend(loc='lower right',fontsize=7.2,handlelength=1.1,labelspacing=0.15,borderaxespad=0.2,framealpha=1,edgecolor=GRID)
ax=axs[2]; rj=np.random.RandomState(1)
for k,(col,mk,lab) in enumerate(((OR,'s','best_score_'),(GREEN,'o','nested'),('#7F8793','^','fresh'))):
    ax.scatter(k+rj.uniform(-0.25,0.25,12),W[:,k],s=6,color=col,marker=mk,lw=0,alpha=0.8); ax.plot([k-0.34,k+0.34],[W[:,k].mean()]*2,color=NAVY,lw=1.4)
ax.set_xticks(range(3)); ax.set_xticklabels(['inner\nCV','nested\nCV','fresh\ndata'],fontsize=7.6); ax.set_xlim(-0.6,2.6); ax.set_ylim(0.94,1.0); ax.set_ylabel('accuracy'); ax.set_title('(c) 12 re-splits',fontsize=8,color=NAVY); grid(ax)
NUM['winner_means']=[round(float(x),4) for x in W.mean(axis=0)]
sv(fig,'f568.png',470)

# ======== 56.8-2 halving: the paths of 72 candidates
hp=np.array(cs['halving_path']); its=sorted(set(hp[:,0].astype(int)))
fig,axs=plt.subplots(1,2,figsize=(4.8,1.75),dpi=240,gridspec_kw={'width_ratios':[1.5,1],'wspace':0.3})
ax=axs[0]; key=lambda r_:(round(r_[3],9),round(r_[4],12)); paths={}
for r_ in hp: paths.setdefault(key(r_),[]).append((int(r_[1]),r_[2],int(r_[0])))
last=max(its); rj=np.random.RandomState(0)
for pth in paths.values():
    pth.sort(); L_=pth[-1][2]; col=GREEN if L_==last else (BLUE if L_>=last-1 else '#9AA3AF')
    xs_=[a for a,_,_ in pth]; ys_=[b_ for _,b_,_ in pth]
    ax.plot(xs_,ys_,'-',color=col,lw=1.2 if col!='#9AA3AF' else 0.6,alpha=0.85,zorder=3 if col==GREEN else 2)
    ax.plot(xs_[-1],ys_[-1],'o',color=col,ms=2.6 if col=='#9AA3AF' else 3.2,mec='none',zorder=4)
res_=sorted(set(hp[:,1].astype(int))); nc=[int(sum(hp[:,0]==i)) for i in its]
ax.set_xscale('log'); ax.set_xticks(res_); ax.set_xticklabels([str(x) for x in res_]); ax.minorticks_off(); ax.set_xlabel('rows used (resource)'); ax.set_ylabel('mean 5-fold accuracy'); ax.set_ylim(0.6,1.02); grid(ax)
for x_,n_ in zip(res_,nc): ax.text(x_,1.035,str(n_),ha='center',fontsize=7.8,color=PUR,clip_on=False)
ax.text(res_[0]*0.78,1.035,'$n_k$:',ha='right',fontsize=7.8,color=PUR,clip_on=False)
ax=axs[1]; cost=[n_*r_ for n_,r_ in zip(nc,res_)]; ax.bar(range(len(cost)),cost,color=PUR,width=0.65,edgecolor=NAVY,lw=0.5)
ax.set_xticks(range(len(cost))); ax.set_xticklabels([str(i) for i in range(len(cost))]); ax.set_xlabel('rung $k$'); ax.set_ylabel('$n_k\\,r_k$ (rows)'); ax.set_ylim(0,2700); grid(ax)
NUM['halving_costs']=[int(c_) for c_ in cost]; NUM['halving_total']=int(sum(cost)); NUM['grid_total']=72*426
ax.text(0.5,0.9,'total %d rows\nfull grid: %d'%(sum(cost),72*426),ha='center',fontsize=7.8,color=NAVY,transform=ax.transAxes,bbox=dict(fc='white',ec='none',pad=1.5))
sv(fig,'f568b.png',460)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print(json.dumps(CHK)); print(NUM)

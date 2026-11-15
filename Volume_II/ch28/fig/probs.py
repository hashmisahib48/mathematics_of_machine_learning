# Problem figures of Chapter 28. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
os.environ['OMP_NUM_THREADS']='1'
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import load_breast_cancer, make_classification, make_friedman1
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from hist_scratch import HistStump
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v: [round(float(u),4) for u in v]

# 28A.1 binning by hand
x1=np.array([3,5,6,8,9,12,14,20.]); thr=np.percentile(x1,[25,50,75],method='averaged_inverted_cdf'); b1=np.searchsorted(thr,x1)
m=HGBC(max_iter=1,max_leaf_nodes=2,min_samples_leaf=1,max_bins=4,early_stopping=False).fit(x1.reshape(-1,1),[0,1]*4)
ref=m._bin_mapper.bin_thresholds_[0]; assert np.allclose(ref,thr)
A['28A.1']={'thresholds':r4(thr),'bins':[int(v) for v in b1]}
fig,ax=small(2.4,1.3)
ax.scatter(x1,np.zeros(8),s=30,color=BLUE,zorder=5,edgecolor='white',lw=0.7)
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_ylim(-0.5,0.5); ax.set_xticks([0,5,10,15,20]); ax.set_xlabel('$x$'); sv(fig,'p_28A1.png')

# 28A.2 best split from a histogram
G=np.array([3,-1,-4,2.]); H=np.array([2,2,2,2.]); lam=1.0
GL,HL=np.cumsum(G)[:-1],np.cumsum(H)[:-1]; GR,HR=G.sum()-GL,H.sum()-HL
gain=GL**2/(HL+lam)+GR**2/(HR+lam)-G.sum()**2/(H.sum()+lam); k=int(np.argmax(gain))
A['28A.2']={'gain':r4(gain),'best_cut_after_bin':k,'wL':round(float(-GL[k]/(HL[k]+lam)),4),'wR':round(float(-GR[k]/(HR[k]+lam)),4)}
fig,ax=small(); ax.bar(range(4),G,color=[BLUE if v>0 else OR for v in G],width=0.6,zorder=3); ax.axhline(0,color=GREY,lw=0.8)
ax.set_xticks(range(4)); ax.set_xlabel('bin'); ax.set_ylabel('$G_b$'); ax.set_ylim(-5,4.2); grid(ax); sv(fig,'p_28A2.png')

# 28A.3 histogram subtraction
Gp=np.array([2.5,-1.5,0.5,-0.5]); Hp=np.array([1.5,1.2,1.0,1.3]); Gs=np.array([1.0,-0.5,0.0,-0.5]); Hs=np.array([0.6,0.4,0.0,0.5])
Gl,Hl=Gp-Gs,Hp-Hs; w=float(-Gl.sum()/(Hl.sum()+1.0))
A['28A.3']={'G_larger':r4(Gl),'H_larger':r4(Hl),'sum_G':round(float(Gl.sum()),4),'sum_H':round(float(Hl.sum()),4),'leaf_value_lambda1':round(w,4)}
fig,ax=small(); xs=np.arange(4); ax.bar(xs-0.2,Gp,0.38,color=BLUE,label='parent',zorder=3); ax.bar(xs+0.2,Gs,0.38,color=OR,label='smaller child',zorder=3)
ax.axhline(0,color=GREY,lw=0.8); ax.set_xticks(xs); ax.set_xlabel('bin'); ax.set_ylabel('$G_b$'); ax.set_ylim(-2.2,4.2); ax.legend(loc='upper right',fontsize=9,borderaxespad=0.1); grid(ax); sv(fig,'p_28A3.png')

# 28A.4 leaf value with L2
r=np.array([3,5,4,8.]); lam4,eta=4.0,0.1; w0=r.sum()/len(r); w4=r.sum()/(len(r)+lam4)
A['28A.4']={'w_no_l2':round(float(w0),4),'w_l2':round(float(w4),4),'step':round(float(eta*w4),4),'ratio':round(float(w4/w0),4),'n_for_90pct':int(0.9*lam4/0.1)}
fig,ax=small(); ax.bar(range(4),r,color=BLUE,width=0.6,zorder=3); ax.axhline(w4,color=GREEN,lw=1.4,ls='--'); ax.axhline(w0,color=RED,lw=1.2,ls=':')
ax.plot([],[],color=RED,ls=':',label='mean'); ax.plot([],[],color=GREEN,ls='--',label='$w$, $\\lambda=4$'); ax.legend(loc='upper left',fontsize=9,borderaxespad=0.1)
ax.set_xticks(range(4)); ax.set_xticklabels(['1','2','3','4']); ax.set_xlabel('sample'); ax.set_ylabel('residual $r_i$'); ax.set_ylim(0,10.5); grid(ax); sv(fig,'p_28A4.png')

# 28B.1 number of bins per column
rng=np.random.default_rng(0)
cols=np.c_[rng.normal(size=1000),rng.integers(0,30,1000).astype(float),np.where(rng.random(1000)<0.6,0.0,rng.normal(size=1000))]
m=HGBC(max_iter=1,early_stopping=False).fit(cols,rng.integers(0,2,1000)); nb=[int(v) for v in m._bin_mapper.n_bins_non_missing_]
A['28B.1']={'bins':nb}
fig,ax=small(); ax.bar(range(3),nb,color=[BLUE,OR,GREEN],width=0.6,zorder=3)
for i,v in enumerate(nb): ax.text(i,v+6,str(v),ha='center',fontsize=9,color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(['normal','30 levels','60% zeros'],fontsize=8.6); ax.set_ylabel('bins'); ax.set_ylim(0,300); grid(ax); sv(fig,'p_28B1.png')

# 28B.2 scratch stump against scikit-learn with 16 bins
X,y=load_breast_cancer(return_X_y=True); p=y.mean()
mine=HistStump(max_bins=16,l2=5.0).fit(X,p-y,np.full(len(y),p*(1-p)))
ref=HGBC(max_iter=1,max_leaf_nodes=2,max_bins=16,l2_regularization=5.0,learning_rate=1.0,early_stopping=False).fit(X,y)
n=ref._predictors[0][0].nodes[0]
thr_mine=float(np.append(mine.thr_[mine.j_],np.inf)[mine.k_])
A['28B.2']={'mine':{'gain':round(float(mine.gain_),4),'feature':int(mine.j_),'thr':round(thr_mine,4)},'sklearn':{'gain':round(float(n['gain']),4),'feature':int(n['feature_idx']),'thr':round(float(n['num_threshold']),4)}}
fig,ax=small(); gs=[]; 
for B_ in (4,8,16,64,255):
    s=HistStump(max_bins=B_,l2=5.0).fit(X,p-y,np.full(len(y),p*(1-p))); gs.append(float(s.gain_))
A['28B.2']['gain_by_bins']={str(b_):round(g_,2) for b_,g_ in zip((4,8,16,64,255),gs)}
ax.plot(range(5),gs,'-o',color=PUR,ms=4,lw=1.4); ax.set_xticks(range(5)); ax.set_xticklabels(['4','8','16','64','255'],fontsize=9); ax.set_xlabel('max_bins'); ax.set_ylabel('root gain'); grid(ax); sv(fig,'p_28B2.png')

# 28B.3 patience of early stopping
Xc,yc=make_classification(12000,20,n_informative=6,flip_y=0.1,random_state=0); res={}
for nn in (3,10,30):
    mm=HGBC(max_iter=500,n_iter_no_change=nn,random_state=0).fit(Xc,yc); v=-mm.validation_score_
    res[nn]={'trees':int(mm.n_iter_),'best_tree':int(np.argmin(v)),'val_logloss':round(float(v.min()),4)}
A['28B.3']=res
fig,ax=small(); ax.plot(range(3),[res[k]['trees'] for k in (3,10,30)],'-o',color=PUR,ms=4,lw=1.4); ax.set_xticks(range(3)); ax.set_xticklabels(['3','10','30']); ax.set_xlim(-0.3,2.3)
ax.set_xlabel('n_iter_no_change'); ax.set_ylabel('trees kept'); grid(ax); sv(fig,'p_28B3.png')

# 28C.1 monotonic constraint
rng=np.random.default_rng(0); xm=rng.uniform(0,10,(400,1)); ym=xm[:,0]+rng.normal(0,2.5,400); gr=np.linspace(0,10,500).reshape(-1,1)
fits={str(c):HGBR(monotonic_cst=c,random_state=0).fit(xm,ym) for c in (None,[1])}
A['28C.1']={'falling_steps':{c:int((np.diff(f.predict(gr))<-1e-9).sum()) for c,f in fits.items()}}
fig,ax=small(); ax.scatter(xm[:,0],ym,s=3,color=GREY,alpha=0.5,zorder=1); ax.plot(gr[:,0],fits['None'].predict(gr),color=OR,lw=1.1,label='free'); ax.plot(gr[:,0],fits['[1]'].predict(gr),color=GREEN,lw=1.5,label='non-decreasing')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.legend(loc='upper left',fontsize=9,borderaxespad=0.1); grid(ax); sv(fig,'p_28C1.png')

# 28C.2 leaves against n
ns=[30,36,39,40,41,50,80,120]; lv={}
for nn in ns:
    Xs,ys=make_friedman1(nn,n_features=5,random_state=0); mm=HGBR(max_iter=1,early_stopping=False).fit(Xs,ys); lv[nn]=int(mm._predictors[0][0].get_n_leaf_nodes())
A['28C.2']=lv
fig,ax=small(); ax.step(ns,[lv[k] for k in ns],where='post',color=PUR,lw=1.5); ax.plot(ns,[lv[k] for k in ns],'o',color=PUR,ms=3.5); ax.set_xlabel('rows $n$'); ax.set_ylabel('leaves of tree 1'); grid(ax); sv(fig,'p_28C2.png')

# 28C.3 the timing claim
with open('verify/speed.json') as fh: sp=json.load(fh)
fig,ax=small()
for key,lab,col,ls in (('gbc','GradientBoosting',OR,'-'),('hgbc','HistGradientBoosting',GREEN,'-')):
    pts=[(n_,t_) for n_,t_ in zip(sp['n'],sp[key]) if t_ is not None]; ax.plot(*zip(*pts),'-o',color=col,ms=3.5,lw=1.4,label=lab)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('rows $n$'); ax.set_ylabel('seconds'); ax.set_ylim(0.15,400); ax.legend(loc='upper left',fontsize=8.6,borderaxespad=0.1); grid(ax); sv(fig,'p_28C3.png')
A['28C.3']={'ratio_at_n':{str(n_):round(a/b,1) for n_,a,b in zip(sp['n'],sp['gbc'],sp['hgbc']) if a is not None}}

with open('verify/problem_answers.json','w') as fh: json.dump(A,fh,indent=1)
with open('verify/probfig_textsize.json','w') as fh: json.dump(CHK,fh,indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

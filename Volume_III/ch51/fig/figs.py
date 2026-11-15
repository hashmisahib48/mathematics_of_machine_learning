# All figures of Chapter 51. Run: python3 fig/figs.py  (PNGs into ch51/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from threadpoolctl import threadpool_limits; threadpool_limits(1)
from sklearn.exceptions import ConvergenceWarning; warnings.simplefilter('ignore',ConvergenceWarning)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from holes import split, punch, HOLE, NAMES
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':8.5,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
Xtr,Xte,ytr,yte=split(); sd=Xtr.std(0)

# ======== 51.0-1 opener: two ways to fill the blanks of s2
j1,j2=4,5; D=Xtr[:,[j1,j2]]/Xtr[:,[j1,j2]].std(0); rng=np.random.RandomState(1); hid=rng.uniform(size=len(D))<0.25
A=D.copy(); A[hid,1]=np.nan
fm=SimpleImputer().fit_transform(A); fi=IterativeImputer(random_state=0).fit_transform(A)
fig,axs=plt.subplots(1,2,figsize=(4.5,1.85),dpi=240,sharey=True)
for ax,F,col,mk,tt,key in ((axs[0],fm,OR,'s','mean fill','mean'),(axs[1],fi,GREEN,'^','regression fill (iterative)','iter')):
    ax.scatter(D[~hid,0],D[~hid,1],s=5,color=BLUE,alpha=0.55,lw=0,label='observed')
    ax.scatter(D[hid,0],D[hid,1],s=9,facecolor='none',edgecolor=GREY,lw=0.6,label='hidden truth')
    ax.scatter(F[hid,0],F[hid,1],s=7,marker=mk,color=col,lw=0,label='filled value')
    e=float(np.sqrt(np.mean((F[hid,1]-D[hid,1])**2))); NUM['open_rmse_'+key]=e
    ax.set_title(tt,fontsize=9,color=NAVY); ax.set_xlabel('$s_1$ (standardised)'); ax.text(0.04,0.96,f'error {e:.2f} sd',transform=ax.transAxes,fontsize=8.5,va='top',color=RED); grid(ax)
    ax.set_xlim(-2.4,4.2); ax.set_ylim(-2.6,4.6)
axs[0].set_ylabel('$s_2$ (standardised)'); axs[1].legend(loc='lower right',handletextpad=0.1,borderaxespad=0.2,fontsize=8.5)
NUM['open']={'n_hidden':int(hid.sum()),'n':len(D),'corr':float(np.corrcoef(D.T)[0,1])}
fig.tight_layout(w_pad=0.8); sv(fig,'f510.png',370)

# ======== 51.2-1 three mechanisms
fig,axs=plt.subplots(1,3,figsize=(4.5,1.6),dpi=240,sharey=True)
col=2; age=Xtr[:,0]/sd[0]; bmi=Xtr[:,col]/sd[col]
for ax,kind,tt in zip(axs,('MCAR','MAR','MNAR'),('MCAR: constant chance','MAR: depends on age','MNAR: depends on bmi')):
    Am=punch(Xtr,kind,0.2,3); m=np.isnan(Am[:,col])
    ax.scatter(age[~m],bmi[~m],s=4,color=BLUE,alpha=0.5,lw=0,label='observed'); ax.scatter(age[m],bmi[m],s=11,marker='x',color=RED,lw=0.8,label='blank')
    ax.set_title(tt,fontsize=8.5,color=NAVY); ax.set_xlabel('age (sd)'); grid(ax)
    NUM['mech_'+kind]={'blank':int(m.sum()),'mean_age_blank':float(age[m].mean()),'mean_age_all':float(age.mean()),'mean_bmi_observed':float(bmi[~m].mean()),'mean_bmi_blank':float(bmi[m].mean())}
axs[0].set_ylabel('bmi (sd)'); h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=2,fontsize=8.5,handletextpad=0.1,columnspacing=1.5,bbox_to_anchor=(0.55,-0.03),frameon=False)
fig.tight_layout(w_pad=0.5,rect=(0,0.08,1,1)); sv(fig,'f512.png',365)

# ======== 51.7-1 flow chart
fig,ax=plt.subplots(figsize=(4.4,2.1),dpi=240); ax.axis('off'); ax.set_xlim(-0.1,12.1); ax.set_ylim(0,6.3)
ax.text(0.0,6.0,'fit, then transform',fontsize=9,color=NAVY,weight='bold',va='center')
box(ax,0,3.5,2.9,1.9,'training\ntable with NaN',fs=8.5); box(ax,4.0,3.5,4.0,1.9,'fit: learn\nstatistics_, rows\nor regressions',fs=8.5,fc='#E6F4EC',ec=GREEN); box(ax,9.0,3.5,3.0,1.9,'transform\nany rows',fs=8.5)
arr(ax,(2.95,4.45),(3.95,4.45),c=NAVY); arr(ax,(8.05,4.45),(8.95,4.45),c=NAVY)
ax.text(0.0,2.9,'IterativeImputer, one round',fontsize=9,color=NAVY,weight='bold',va='center')
box(ax,0,0.2,2.9,1.9,'fill each blank\nwith the mean\n(first round)',fs=8.5); box(ax,4.0,0.2,4.0,1.9,'for each feature j:\nregress $x_j$ on the rest\n(rows where observed)',fs=8.5,fc='#E6F4EC',ec=GREEN); box(ax,9.0,0.2,3.0,1.9,'replace its\nblanks by the\nprediction',fs=8.5,fc=PINK,ec=RED)
arr(ax,(2.95,1.15),(3.95,1.15),c=NAVY); arr(ax,(8.05,1.15),(8.95,1.15),c=NAVY)
ax.annotate('',xy=(6.0,2.15),xytext=(10.5,2.15),arrowprops=dict(arrowstyle='-|>',color=PUR,lw=1.0,connectionstyle='arc3,rad=0'))
ax.text(8.3,2.5,'repeat until the change < tol',fontsize=8.5,color=PUR,ha='center',va='center')
sv(fig,'f571.png',300)

# ======== 51.10-1 hyperparameters: k, rounds, variance
MC=12; ks=[1,2,3,5,8,12,20,40]
def nrmse(F,Xtrue,m): return float(np.mean([np.sqrt(np.mean((F[m[:,j],j]-Xtrue[m[:,j],j])**2))/Xtrue[:,j].std() for j in HOLE]))
res={'mean':[], 'uniform':{k:[] for k in ks}, 'distance':{k:[] for k in ks}}; rounds=list(range(0,11)); itr={r:[] for r in rounds}; chg=[]
for s in range(MC):
    Am=punch(Xtr,'MCAR',0.2,s); m=np.isnan(Am)
    res['mean'].append(nrmse(SimpleImputer().fit_transform(Am),Xtr,m))
    for k in ks:
        for w in ('uniform','distance'): res[w][k].append(nrmse(KNNImputer(n_neighbors=k,weights=w).fit_transform(Am),Xtr,m))
    for r in rounds: itr[r].append(nrmse(IterativeImputer(max_iter=r,random_state=0).fit_transform(Am),Xtr,m))
fig,axs=plt.subplots(1,3,figsize=(4.7,1.6),dpi=240,gridspec_kw={'width_ratios':[1,1,1.25]})
ax=axs[0]
for w,c_,mk in (('uniform',BLUE,'o'),('distance',GREEN,'s')): ax.plot(ks,[np.mean(res[w][k]) for k in ks],color=c_,marker=mk,ms=3,lw=1.1,label=w)
ax.axhline(np.mean(res['mean']),color=OR,lw=1.2,ls='--'); ax.text(40,np.mean(res['mean'])+0.012,'mean fill',color=OR,fontsize=8.5,va='bottom',ha='right')
ax.set_xscale('log'); ax.set_xticks([1,3,10,40]); ax.set_xticklabels(['1','3','10','40']); ax.minorticks_off(); ax.set_xlabel('neighbours $k$'); ax.set_ylabel('fill error (sd)'); ax.legend(loc='lower right',handlelength=1.3,borderaxespad=0.1,fontsize=8.5); grid(ax); ax.set_ylim(0.3,1.1)
ax=axs[1]; ax.plot(rounds,[np.mean(itr[r]) for r in rounds],color=PUR,marker='o',ms=3,lw=1.1); ax.axhline(np.mean(res['mean']),color=OR,lw=1.2,ls='--'); ax.set_xlabel('rounds (max_iter)'); ax.set_xticks([0,2,4,6,8,10]); grid(ax); ax.set_ylim(0.3,1.1)
ax.text(10,0.94,'0 = mean fill',fontsize=8.5,color=GREY,ha='right',va='top')
ax.set_yticklabels([])
ax=axs[2]; V=json.load(open('verify/prose_numbers.json'))['var_ratio']; names=['mean','kNN','iter','post']; vals=[V['mean'],V['knn'],V['iterative'],V['posterior']]
ax.bar(range(4),vals,color=[OR,BLUE,PUR,GREEN],width=0.65); ax.axhline(1,color=GREY,lw=0.8,ls='--'); ax.set_xticks(range(4)); ax.set_xticklabels(names,fontsize=8.5); ax.set_ylim(0.6,1.15); ax.set_ylabel('var. filled / true'); grid(ax)
for i_,v in enumerate(vals): ax.text(i_,v+0.01,f'{v:.2f}',ha='center',fontsize=8.5,color=NAVY)
NUM['hyper']={'k':ks,'uniform':[float(np.mean(res['uniform'][k])) for k in ks],'distance':[float(np.mean(res['distance'][k])) for k in ks],'mean_fill':float(np.mean(res['mean'])),'rounds':rounds,'iter':[float(np.mean(itr[r])) for r in rounds]}
fig.tight_layout(w_pad=0.5); sv(fig,'f5101.png',385)

# ======== 51.11-1 case study
R=json.load(open('verify/case.json')); rates=[0.1,0.2,0.3]
fig,axs=plt.subplots(1,3,figsize=(4.7,1.8),dpi=240,sharey=True)
nm={'mean':'mean','mean+ind.':'mean+ind','KNN':'KNN k=5','iterative':'iterative','native NaN':'native NaN'}
sty={'mean':(OR,'s','-'),'mean+ind.':(OR,'o','--'),'KNN':(BLUE,'^','-'),'iterative':(GREEN,'o','-'),'native NaN':(PUR,'D','-')}
for ax,kind in zip(axs,('MCAR','MAR','MNAR')):
    for n,(c_,mk,ls) in sty.items(): ax.plot(rates,[np.mean(R[f'{kind}_{q}'][nm[n]]) for q in rates],color=c_,marker=mk,ms=3.2,lw=1.1,ls=ls,label=n)
    ax.axhline(0.359,color=GREY,lw=0.9,ls=':'); ax.set_title(kind,fontsize=9,color=NAVY); ax.set_xticks(rates); ax.set_xticklabels(['10%','20%','30%']); ax.set_xlabel('share of blank cells'); grid(ax); ax.set_ylim(0.24,0.37)
axs[0].set_ylabel('test $R^2$'); axs[0].text(0.1,0.361,'complete data',fontsize=8.5,color=GREY,va='bottom')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower center',ncol=5,fontsize=8.5,handlelength=1.5,columnspacing=0.9,handletextpad=0.3,bbox_to_anchor=(0.5,-0.02),frameon=False)
fig.tight_layout(w_pad=0.5,rect=(0,0.09,1,1)); sv(fig,'f5111.png',385)
json.dump({'num':NUM,'font_check':CHK},open('verify/fig_numbers.json','w'),indent=1)
for k,v in CHK.items(): print(k,v)
print(json.dumps(NUM['open_rmse_mean'])); print(NUM['open_rmse_iter']); print({k:v for k,v in NUM.items() if k.startswith('mech')})

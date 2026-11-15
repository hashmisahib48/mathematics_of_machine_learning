# Problem figures of Chapter 51. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from threadpoolctl import threadpool_limits; threadpool_limits(1)
from sklearn.exceptions import ConvergenceWarning; warnings.simplefilter('ignore',ConvergenceWarning)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics.pairwise import nan_euclidean_distances
from holes import split, punch, HOLE
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.0,h=0.95): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*106,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]
Xtr,Xte,ytr,yte=split()

# 51A.1 nan-Euclidean distance by hand
nan=np.nan; M=np.array([[2,nan,6],[4,5,nan],[1,3,9],[nan,6,5]])
d=nan_euclidean_distances(M[:1],M)[0]
k2=KNNImputer(n_neighbors=2).fit_transform(M)[0,1]; k1=KNNImputer(n_neighbors=1).fit_transform(M)[0,1]
A['51A.1']={'d_ab':round(float(d[1]),3),'d_ac':round(float(d[2]),3),'d_ad':round(float(d[3]),3),'k1':float(k1),'k2':float(k2)}
fig,ax=small(2.0,0.95); ax.imshow(~np.isnan(M),cmap='Blues',vmin=0,vmax=2.0,aspect='auto')
for i in range(4):
    for j in range(3): ax.text(j,i,'?' if np.isnan(M[i,j]) else f'{int(M[i,j])}',ha='center',va='center',fontsize=9.5,color=RED if np.isnan(M[i,j]) else NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(['col 1','col 2','col 3']); ax.set_yticks(range(4)); ax.set_yticklabels(['a','b','c','d']); ax.spines[:].set_visible(False); ax.tick_params(length=0); sv(fig,'p_51A1.png')

# 51A.2 variance after mean fill
v=np.array([2,4,6,8.]); mo=v.mean(); so=v.var(ddof=0); filled=np.append(v,[mo,mo])
A['51A.2']={'mean_observed':float(mo),'var_observed_divisor_n_o':float(so),'var_filled_divisor_n':round(float(filled.var()),4),'ratio':round(float(filled.var()/so),4)}
fig,ax=small(2.0,0.95); ax.scatter(v,np.zeros(4),s=22,color=BLUE,zorder=3); ax.scatter([mo,mo],[0.0,0.0],s=40,marker='s',facecolor='none',edgecolor=OR,lw=1.2,zorder=2)
ax.axvline(mo,color=GREEN,lw=0.9,ls='--'); ax.text(mo+0.2,0.55,'mean 5',color=GREEN,fontsize=9); ax.set_ylim(-0.7,1.0); ax.set_yticks([]); ax.set_xticks([2,4,6,8]); ax.spines['left'].set_visible(False); ax.set_xlim(1,9.5); grid(ax); sv(fig,'p_51A2.png')

# 51B.1 kNN fill error
def nrmse(F,Xt,m): return float(np.mean([np.sqrt(np.mean((F[m[:,j],j]-Xt[m[:,j],j])**2))/Xt[:,j].std() for j in HOLE]))
Am=punch(Xtr,'MCAR',0.2,0); mk_=np.isnan(Am); ks=[1,3,10,30]; res={w:[nrmse(KNNImputer(n_neighbors=k,weights=w).fit_transform(Am),Xtr,mk_) for k in ks] for w in ('uniform','distance')}
A['51B.1']={'k':ks,'uniform':r4(res['uniform']),'distance':r4(res['distance']),'mean_fill':round(nrmse(SimpleImputer().fit_transform(Am),Xtr,mk_),4),'n_blank_cells':int(mk_.sum())}
fig,ax=small(2.0,0.95); ax.plot(ks,res['uniform'],color=BLUE,marker='o',ms=3,lw=1.1); ax.plot(ks,res['distance'],color=GREEN,marker='s',ms=3,lw=1.1,ls='--'); ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off(); ax.set_xlabel('neighbours $k$'); ax.set_ylabel('error (sd)'); grid(ax); sv(fig,'p_51B1.png')

# 51B.2 posterior draws
det=IterativeImputer(random_state=0).fit_transform(Am); dr=[IterativeImputer(sample_posterior=True,random_state=s).fit_transform(Am) for s in range(5)]
e_det=nrmse(det,Xtr,mk_); e_draw=[nrmse(F,Xtr,mk_) for F in dr]; e_avg=nrmse(np.mean(dr,axis=0),Xtr,mk_)
A['51B.2']={'deterministic':round(e_det,4),'single_draws':r4(e_draw),'mean_single_draw':round(float(np.mean(e_draw)),4),'average_of_5_draws':round(e_avg,4)}
fig,ax=small(2.0,0.95); xs=np.arange(3); ax.bar(xs,[e_det,float(np.mean(e_draw)),e_avg],color=[PUR,GREEN,BLUE],width=0.6); ax.set_xticks(xs); ax.set_xticklabels(['plain','one draw','mean of 5'],fontsize=9.5); ax.set_ylabel('error (sd)'); ax.set_ylim(0,1.0); grid(ax)
for i_,v_ in enumerate([e_det,float(np.mean(e_draw)),e_avg]): ax.text(i_,v_+0.03,f'{v_:.2f}',ha='center',fontsize=9.5,color=NAVY)
sv(fig,'p_51B2.png')

# 51C.1 test ordered only when suspected
rng=np.random.RandomState(2); v=np.concatenate([rng.normal(1.2,0.5,14),rng.normal(-0.2,0.25,0)]); fig,ax=small(2.0,0.95)
ax.scatter(v,np.full(len(v),0.0),s=12,color=BLUE,zorder=3); xb=rng.normal(-0.9,0.35,12); ax.scatter(xb,np.full(len(xb),0.0),s=14,marker='x',color=RED,lw=0.8,zorder=3)
ax.text(1.2,0.55,'ordered',color=BLUE,fontsize=9,ha='center'); ax.text(-0.9,0.55,'blank',color=RED,fontsize=9,ha='center'); ax.set_ylim(-0.6,1.0); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('true test value'); grid(ax); sv(fig,'p_51C1.png')

# 51C.2 a column that is nearly empty
fig,ax=small(2.0,0.95); names=['age','bmi','bp','test']; fr=[0.0,0.03,0.05,0.95]; ax.bar(range(4),fr,color=[BLUE,BLUE,BLUE,RED],width=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylabel('blank'); ax.set_ylim(0,1.1); ax.set_yticks([0,0.5,1]); grid(ax); sv(fig,'p_51C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/problem_font_check.json','w'),indent=1)
print(json.dumps(A,indent=0)); print(CHK)

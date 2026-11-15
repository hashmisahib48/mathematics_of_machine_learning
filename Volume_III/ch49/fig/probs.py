# Problem figures of Chapter 49. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy import sparse, stats
from scipy.stats import skew
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.15): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]

# 49A.1 new values against a fitted scaler
tr=np.array([2,4,6,8,30.]).reshape(-1,1); new=np.array([[12.],[40.]])
st=StandardScaler().fit(tr); mm=MinMaxScaler().fit(tr); mc=MinMaxScaler(clip=True).fit(tr); rb=RobustScaler().fit(tr)
A['49A.1']={'train':tr.ravel().tolist(),'new':new.ravel().tolist(),'mean':float(st.mean_[0]),'sd':round(float(st.scale_[0]),4),'min':2,'max':30,'median':float(rb.center_[0]),'iqr':float(rb.scale_[0]),
 'standard':r4(st.transform(new)),'minmax':r4(mm.transform(new)),'minmax_clip':r4(mc.transform(new)),'robust':r4(rb.transform(new))}
fig,ax=small(2.4,0.9); ax.plot(tr.ravel(),np.zeros(5),'o',color=BLUE,ms=5); ax.plot(new.ravel(),np.zeros(2),'*',color=RED,ms=8)
ax.axvspan(4,8,ymin=0.3,ymax=0.7,color=LB); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(0,42); ax.set_ylim(-1,1); ax.set_xticks([0,10,20,30,40]); ax.set_xlabel('value'); ax.text(6,0.55,'Q1 to Q3',fontsize=9,ha='center',color=NAVY); sv(fig,'p_49A1.png')

# 49A.2 Yeo-Johnson by hand
xs=np.array([-3,-1,0,1,3.]); y0=stats.yeojohnson(xs,lmbda=0.0); y2=stats.yeojohnson(xs,lmbda=2.0)
A['49A.2']={'x':xs.tolist(),'lambda0':r4(y0),'lambda2':r4(y2),'mirror_ok':bool(np.allclose(y2,-stats.yeojohnson(-xs,lmbda=0.0)))}
fig,ax=small(2.4,1.0); xg=np.linspace(-3.2,3.2,200)
ax.plot(xg,stats.yeojohnson(xg,lmbda=0.0),color=BLUE,lw=1.3,label='$\\lambda=0$'); ax.plot(xg,stats.yeojohnson(xg,lmbda=2.0),color=PUR,lw=1.3,ls='--',label='$\\lambda=2$')
ax.plot(xs,y0,'o',color=BLUE,ms=3.5); ax.plot(xs,y2,'s',color=PUR,ms=3.5); ax.set_ylim(-8,11); ax.set_xlabel('$x$'); ax.set_ylabel('$\\psi$'); ax.legend(loc='upper left',ncol=2,handlelength=1.3,columnspacing=0.8,handletextpad=0.4,borderaxespad=0.2); grid(ax); sv(fig,'p_49A2.png')

# 49B.1 sparse data
Xd=load_digits().data/16.; S=sparse.csr_matrix(Xd); frac=lambda M:(M.nnz/np.prod(M.shape))
res={'raw':frac(S)}
res['MaxAbsScaler']=frac(MaxAbsScaler().fit_transform(S))
res['StandardScaler(with_mean=False)']=frac(StandardScaler(with_mean=False).fit_transform(S))
res['RobustScaler(with_centering=False)']=frac(RobustScaler(with_centering=False).fit_transform(S))
dense_std=StandardScaler().fit_transform(Xd); res['StandardScaler, dense']=float((dense_std!=0).mean())
msg={}
for nm,f in (('StandardScaler()',lambda:StandardScaler().fit(S)),('MinMaxScaler()',lambda:MinMaxScaler().fit(S))):
    try: f(); msg[nm]='ok'
    except Exception as e: msg[nm]=type(e).__name__
zero_cols=int((np.asarray(S.sum(0)).ravel()==0).sum())
sd_sparse=StandardScaler(with_mean=False).fit(S); A['49B.1']={'nonzero_fraction':{k:round(float(v),4) for k,v in res.items()},'errors':msg,'all_zero_columns':zero_cols,'scale_of_zero_column':float(sd_sparse.scale_[np.asarray(S.sum(0)).ravel()==0][0]),'shape':list(S.shape)}
fig,ax=small(2.4,1.0); labs=['raw','MaxAbs','Std,\nno mean','Std,\ncentred']; vals=[res['raw'],res['MaxAbsScaler'],res['StandardScaler(with_mean=False)'],res['StandardScaler, dense']]
ax.bar(range(4),vals,color=[BLUE,GREEN,GREEN,RED],width=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(labs,fontsize=8.5); ax.set_ylabel('non-zero share'); ax.set_ylim(0,1.12); grid(ax); sv(fig,'p_49B1.png')

# 49B.2 skewness before and after
bc=load_breast_cancer(); Xb=bc.data
sk_raw=skew(Xb,axis=0); sk_yj=skew(PowerTransformer().fit_transform(Xb),axis=0); sk_qn=skew(QuantileTransformer(n_quantiles=200,output_distribution='normal').fit_transform(Xb),axis=0)
pt=PowerTransformer().fit(Xb); j=int(np.abs(sk_raw).argmax())
A['49B.2']={'mean_abs_skew':{'raw':round(float(np.abs(sk_raw).mean()),3),'yeo_johnson':round(float(np.abs(sk_yj).mean()),3),'quantile_normal':round(float(np.abs(sk_qn).mean()),3)},
 'max_abs_skew_raw':round(float(np.abs(sk_raw).max()),3),'most_skewed':str(bc.feature_names[j]),'lambda_most_skewed':round(float(pt.lambdas_[j]),3),'skew_after_yj_of_it':round(float(sk_yj[j]),3),
 'n_features_raw_skew_above_1':int((np.abs(sk_raw)>1).sum()),'n_features_yj_skew_above_0.5':int((np.abs(sk_yj)>0.5).sum()),'max_abs_skew_yj':round(float(np.abs(sk_yj).max()),3),'max_abs_skew_qn':round(float(np.abs(sk_qn).max()),3)}
fig,ax=small(2.4,1.1); o=np.argsort(np.abs(sk_raw))
ax.plot(np.arange(30),np.abs(sk_raw)[o],'o',color=RED,ms=3,label='raw'); ax.plot(np.arange(30),np.abs(sk_yj)[o],'s',color=BLUE,ms=3,label='Yeo-Johnson'); ax.plot(np.arange(30),np.abs(sk_qn)[o],'^',color=GREEN,ms=3,label='quantile')
ax.set_xlabel('feature (sorted by raw skew)'); ax.set_ylabel('|skewness|'); ax.set_xticks([]); ax.legend(loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3,handlelength=0.6,handletextpad=0.2,columnspacing=0.6,fontsize=8.5,borderaxespad=0.1,frameon=False); grid(ax); sv(fig,'p_49B2.png')

# 49C.1 scaling before cross-validation: leaky minus honest, 20 subsets of 50 rows
Xt,yt=load_breast_cancer(return_X_y=True); out={}; dif={}
mks=(('Standard',lambda:StandardScaler()),('Quantile',lambda:QuantileTransformer(n_quantiles=30,output_distribution='normal')),('Power',lambda:PowerTransformer()))
for nm,mk in mks:
    h=cross_val_score(make_pipeline(mk(),SVC()),Xt,yt,cv=5).mean(); l=cross_val_score(SVC(),mk().fit_transform(Xt),yt,cv=5).mean(); out[nm]=[round(float(h),4),round(float(l),4)]
    d=[]
    for sd in range(20):
        idx=np.random.RandomState(sd).permutation(len(yt))[:50]
        d.append(cross_val_score(SVC(),mk().fit_transform(Xt[idx]),yt[idx],cv=5).mean()-cross_val_score(make_pipeline(mk(),SVC()),Xt[idx],yt[idx],cv=5).mean())
    dif[nm]=d
A['49C.1']={'full_data_honest_leaky':out,'subsets_of_50':{k:{'mean':round(float(np.mean(v)),4),'sd':round(float(np.std(v)),4),'min':round(float(np.min(v)),3),'max':round(float(np.max(v)),3),'n_nonzero':int((np.abs(v)>1e-9).sum())} for k,v in dif.items()}}
fig,ax=small(2.4,1.0)
for k,(nm,_) in enumerate(mks):
    jit=np.linspace(-0.22,0.22,20); ax.plot(k+jit,dif[nm],'o',color=BLUE,ms=2.6,alpha=0.8); ax.plot([k-0.3,k+0.3],[np.mean(dif[nm])]*2,color=RED,lw=1.6)
ax.axhline(0,color=GREY,lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(['Standard','Quantile','Power'],fontsize=8.5); ax.set_ylabel('leaky $-$ honest'); ax.set_xlim(-0.6,2.6); grid(ax); sv(fig,'p_49C1.png')

# 49C.2 trees do not care
rng=np.random.RandomState(4); xx=np.sort(rng.lognormal(0,0.9,40)); yy=np.where(xx<1.2,0.0,1.0)+0.25*np.sin(3*np.log(xx))+rng.normal(0,0.12,40)
t1=DecisionTreeRegressor(max_depth=2,random_state=0).fit(xx.reshape(-1,1),yy); t2=DecisionTreeRegressor(max_depth=2,random_state=0).fit(np.log(xx).reshape(-1,1),yy)
same=bool(np.array_equal(t1.predict(xx.reshape(-1,1)),t2.predict(np.log(xx).reshape(-1,1))))
gx=np.linspace(xx.min(),xx.max(),2000); d=(t1.predict(gx.reshape(-1,1))!=t2.predict(np.log(gx).reshape(-1,1))).mean()
A['49C.2']={'training_predictions_identical':same,'share_of_grid_points_that_differ':round(float(d),4),'thresholds_x':r4(np.sort(t1.tree_.threshold[t1.tree_.feature>=0])),'thresholds_exp_logx':r4(np.exp(np.sort(t2.tree_.threshold[t2.tree_.feature>=0])))}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.0),dpi=240,sharey=True)
for ax,f,tt,t in ((axs[0],lambda v:v,'$x$',t1),(axs[1],np.log,'$\\ln x$',t2)):
    ax.plot(f(xx),yy,'o',color=BLUE,ms=2.2); gg=np.linspace(f(xx).min(),f(xx).max(),400); ax.plot(gg,t.predict(gg.reshape(-1,1)),color=GREEN,lw=1.3); ax.set_xlabel(tt); ax.set_xticks([])
axs[0].set_ylabel('$y$'); sv(fig,'p_49C2.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

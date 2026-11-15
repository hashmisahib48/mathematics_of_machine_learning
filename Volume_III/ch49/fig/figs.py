# All figures of Chapter 49. Run: python3 fig/figs.py  (PNGs into ch49/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy.stats import skew, norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from skewdata import make_skewed, make_parcels
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ======== 49.0-1 opener: rulers and the unit of a column
fig,axs=plt.subplots(1,2,figsize=(4.5,2.0),dpi=240,gridspec_kw={'width_ratios':[1,1.05]})
ax=axs[0]; xv=np.array([1,2,3,4,10.]); col=xv.reshape(-1,1)
rows=[('raw $x$',xv),('standard',StandardScaler().fit_transform(col).ravel()),('min-max',MinMaxScaler().fit_transform(col).ravel()),
      ('robust',RobustScaler().fit_transform(col).ravel()),('quantile',QuantileTransformer(n_quantiles=5).fit_transform(col).ravel())]
for k,(nm,v) in enumerate(rows):
    y=len(rows)-1-k; ax.plot([-1.6,10.6],[y,y],color=GRID,lw=0.8,zorder=1)
    ax.plot(v,np.full(5,y),'o',color=BLUE,ms=4.5,zorder=3); ax.plot(v[-1],y,'o',color=RED,ms=5.2,zorder=4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1]); ax.set_xlim(-1.6,10.6); ax.set_ylim(-0.6,4.6); ax.set_xlabel('value after the map'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
ax.set_title('one column, five rulers',fontsize=9,color=NAVY); ax.set_xticks([0,2,4,6,8,10])
# right: accuracy vs unit of the weight column
Xp,yp=make_parcels(600,seed=1); ts=np.arange(-4,1.01,0.5); acc={'knn':[], 'svc':[], 'knn_s':[], 'svc_s':[]}
for t in ts:
    a={k:[] for k in acc}
    for sd in range(6):
        X=Xp.copy(); X[:,1]*=10.0**t; Xa,Xb,ya,yb=train_test_split(X,yp,test_size=0.5,random_state=sd)
        a['knn'].append(KNeighborsClassifier(5).fit(Xa,ya).score(Xb,yb)); a['svc'].append(SVC().fit(Xa,ya).score(Xb,yb))
        a['knn_s'].append(make_pipeline(StandardScaler(),KNeighborsClassifier(5)).fit(Xa,ya).score(Xb,yb)); a['svc_s'].append(make_pipeline(StandardScaler(),SVC()).fit(Xa,ya).score(Xb,yb))
    for k in acc: acc[k].append(float(np.mean(a[k])))
NUM['unit_knob']={'t':ts.tolist(),**{k:[round(v,4) for v in vv] for k,vv in acc.items()}}
ax=axs[1]; ax.plot(ts,acc['knn'],'-s',color=RED,ms=3.2,lw=1.1,label='5-NN raw'); ax.plot(ts,acc['svc'],'-o',color=OR,ms=3.2,lw=1.1,label='SVC raw')
ax.plot(ts,(np.array(acc['knn_s'])+np.array(acc['svc_s']))/2,'-^',color=GREEN,ms=3.2,lw=1.4,label='both scaled')
ax.set_xlabel('weight multiplied by $10^{t}$'); ax.set_ylabel('test accuracy'); ax.set_ylim(0.42,1.04); ax.axvline(0,color=PUR,lw=0.9,ls=':'); ax.axvline(-3,color=PUR,lw=0.9,ls=':')
ax.text(-0.08,1.005,'grams',color=PUR,fontsize=8.5,ha='right'); ax.text(-2.92,1.005,'kilograms',color=PUR,fontsize=8.5,ha='left')
ax.legend(loc='lower left',handlelength=1.3,labelspacing=0.2,fontsize=8,borderaxespad=0.2,handletextpad=0.4,framealpha=1,edgecolor=GRID); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f490.png',410)


# ======== 49.2-1 intuition: five nearest neighbours of one parcel, raw and standardised
Xp,yp=make_parcels(120,seed=1); sc=StandardScaler().fit(Xp); Zp=sc.transform(Xp)
def nn(Xm,q,k=5):
    d=np.sqrt(((Xm-Xm[q])**2).sum(1)); d[q]=np.inf; return np.argsort(d)[:k]
cand=[q for q in range(len(Xp)) if 3.9<Xp[q,0]<4.5 and 350<Xp[q,1]<650]
best=None
for q in cand:
    vr=np.bincount(yp[nn(Xp,q)],minlength=2); vs=np.bincount(yp[nn(Zp,q)],minlength=2)
    if vr.argmax()!=yp[q] and vs.argmax()==yp[q] and vs.max()>=4: best=q; break
q=best if best is not None else cand[0]
NUM['query']={'index':int(q),'km':float(Xp[q,0]),'g':float(Xp[q,1]),'class':int(yp[q]),'votes_raw':np.bincount(yp[nn(Xp,q)],minlength=2).tolist(),'votes_scaled':np.bincount(yp[nn(Zp,q)],minlength=2).tolist(),'km_range_raw_neighbours':[float(Xp[nn(Xp,q),0].min()),float(Xp[nn(Xp,q),0].max())],'g_range_raw_neighbours':[float(Xp[nn(Xp,q),1].min()),float(Xp[nn(Xp,q),1].max())]}
fig,axs=plt.subplots(1,2,figsize=(5.0,1.85),dpi=240)
for ax,M,ttl,xl,yl in ((axs[0],Xp,'raw units','distance (km)','weight (g)'),(axs[1],Zp,'standardised','distance (z-score)','weight (z-score)')):
    for c_,col,mk in ((0,BLUE,'o'),(1,OR,'s')): ax.scatter(M[yp==c_,0],M[yp==c_,1],s=9,color=col,marker=mk,lw=0,alpha=0.75)
    nb=nn(M,q)
    for j in nb: ax.plot([M[q,0],M[j,0]],[M[q,1],M[j,1]],color=GREEN,lw=1.1,zorder=4)
    ax.scatter(M[nb,0],M[nb,1],s=26,facecolor='none',edgecolor=GREEN,lw=1.1,zorder=5)
    ax.scatter(M[q,0],M[q,1],s=44,marker='*',color=RED,zorder=6)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl+': votes %d : %d'%tuple(np.bincount(yp[nb],minlength=2)),fontsize=9,color=NAVY); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f491.png',410)


# ======== 49.8-1 the gallery: one skewed feature with an outlier under every transformation
from sklearn.preprocessing import FunctionTransformer
x=make_skewed()
panels=[('raw',None),('StandardScaler',StandardScaler()),('MinMaxScaler',MinMaxScaler()),('RobustScaler',RobustScaler()),
        ('log1p',FunctionTransformer(np.log1p)),('PowerTransformer',PowerTransformer()),('Quantile, uniform',QuantileTransformer(n_quantiles=200)),('Quantile, normal',QuantileTransformer(n_quantiles=200,output_distribution='normal'))]
fig,axs=plt.subplots(2,4,figsize=(5.0,2.4),dpi=240); G={}
for ax,(nm,tr) in zip(axs.ravel(),panels):
    z=(x if tr is None else tr.fit_transform(x)).ravel(); lo,hi=np.percentile(z,[5,95]); bulk=(hi-lo)/(z.max()-z.min()); sk=skew(z)
    G[nm]={'min':float(z.min()),'max':float(z.max()),'skew':float(sk),'bulk':float(bulk),'median':float(np.median(z))}
    ax.hist(z,bins=30,color=BLUE,alpha=0.85,lw=0); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.set_title(nm,fontsize=8.5,color=NAVY,pad=10); ax.text(0.5,1.02,'skew %.1f, bulk %.2f'%(sk,bulk),transform=ax.transAxes,ha='center',va='bottom',fontsize=7.8,color=GREY)
    ax.tick_params(labelsize=8); ax.locator_params(axis='x',nbins=4)
NUM['gallery']=G
fig.tight_layout(h_pad=0.9,w_pad=0.6); sv(fig,'f493.png',470)


# ======== 49.3-2 the likelihood of lambda and the Yeo-Johnson family
from scipy import stats
x=make_skewed().ravel(); lg=np.log(x).sum(); ls_=np.linspace(-1.2,1.6,141)
def llf(l): z=np.log(x) if l==0 else (x**l-1)/l; return -len(x)/2*np.log(z.var())+(l-1)*lg
ll=np.array([llf(l) for l in ls_]); lhat=PowerTransformer(method='box-cox',standardize=False).fit(x.reshape(-1,1)).lambdas_[0]
NUM['bc_llf']={'lambda_hat':float(lhat),'llf_hat':float(llf(lhat)),'llf_1':float(llf(1.0)),'llf_0':float(llf(1e-9)),'drop_to_1':float(llf(lhat)-llf(1.0)),'drop_to_0':float(llf(lhat)-llf(1e-9)),'chi2_95_half':float(stats.chi2.ppf(0.95,1)/2)}
ci=ls_[ll>=ll.max()-stats.chi2.ppf(0.95,1)/2]; NUM['bc_llf']['ci95']=[float(ci.min()),float(ci.max())]
fig,axs=plt.subplots(1,2,figsize=(5.0,2.0),dpi=240)
ax=axs[0]; rel=ll-ll.max(); ax.plot(ls_,rel,color=BLUE,lw=1.6); ax.plot(lhat,0,'o',color=GREEN,ms=5,zorder=5); ax.axvline(1,color=GREY,lw=0.9,ls=':'); ax.axvline(0,color=GREY,lw=0.9,ls=':')
ax.text(1.04,-60,'$\\lambda=1$:\nno change',fontsize=8,ha='left',va='center',color=GREY); ax.text(-0.04,-200,'$\\lambda=0$: log',fontsize=8,ha='right',color=GREY)
ax.set_ylim(-320,25); ax.set_xlabel('$\\lambda$'); ax.set_ylabel('$\\ell(\\lambda)-\\ell(\\hat{\\lambda})$'); ax.set_title('Box-Cox: $\\hat{\\lambda}=%.2f$'%lhat,fontsize=9,color=NAVY); grid(ax)
ax=axs[1]; xg=np.linspace(-3,3,300)
for l,col,ls2 in ((0,BLUE,'-'),(0.5,GREEN,'--'),(1,GREY,':'),(1.5,OR,'-.'),(2,PUR,'-')):
    ax.plot(xg,stats.yeojohnson(xg,lmbda=l),color=col,lw=1.4,ls=ls2,label='$\\lambda=%g$'%l)
ax.set_xlabel('$x$'); ax.set_ylabel('$\\psi(x;\\lambda)$'); ax.set_ylim(-6,9); ax.set_title('Yeo-Johnson family',fontsize=9,color=NAVY); ax.legend(loc='upper left',handlelength=1.6,labelspacing=0.15,fontsize=8,ncol=2,columnspacing=0.8,handletextpad=0.4,borderaxespad=0.2); grid(ax)
fig.tight_layout(w_pad=1.0); sv(fig,'f494.png',430)


# ======== 49.5-1 the algorithm: fit on the training rows, store, apply
fig,ax=plt.subplots(figsize=(5.0,1.6),dpi=240); ax.set_xlim(-0.1,10.1); ax.set_ylim(0,4.3); ax.axis('off')
box(ax,0.0,3.0,1.7,1.0,'training\nrows $\\mathbf{X}_{\\mathrm{train}}$',fc=LB,ec=BLUE,fs=8.5)
box(ax,2.3,3.0,3.2,1.0,'fit, column by column:\n$c_j,s_j$  |  $\\lambda_j$  |  quantiles',fc=LGREY,ec=GREY,fs=8.5)
box(ax,6.1,3.0,1.8,1.0,'transform\n$\\mathbf{Z}_{\\mathrm{train}}$',fc='#E3F4EA',ec=GREEN,fs=8.5)
box(ax,8.4,3.0,1.6,1.0,'model\n`fit`'.replace('`',''),fc=PINK,ec=RED,fs=8.5)
arr(ax,(1.7,3.5),(2.3,3.5),c=NAVY,lw=1,ms=7); arr(ax,(5.5,3.5),(6.1,3.5),c=NAVY,lw=1,ms=7); arr(ax,(7.9,3.5),(8.4,3.5),c=NAVY,lw=1,ms=7)
box(ax,0.0,0.2,1.7,1.0,'later rows\n$\\mathbf{X}_{\\mathrm{new}}$',fc='#F1F2F4',ec=GREY,fs=8.5)
box(ax,2.3,0.2,3.2,1.0,'stored numbers\nonly, no refitting',fc=LGREY,ec=GREY,fs=8.5)
box(ax,6.1,0.2,1.8,1.0,'transform\n$\\mathbf{Z}_{\\mathrm{new}}$',fc='#E3F4EA',ec=GREEN,fs=8.5)
box(ax,8.4,0.2,1.6,1.0,'model\npredict',fc=PINK,ec=RED,fs=8.5)
arr(ax,(1.7,0.7),(2.3,0.7),c=NAVY,lw=1,ms=7); arr(ax,(5.5,0.7),(6.1,0.7),c=NAVY,lw=1,ms=7); arr(ax,(7.9,0.7),(8.4,0.7),c=NAVY,lw=1,ms=7)
arr(ax,(3.9,3.0),(3.9,1.2),c=PUR,lw=1.3,ms=8); ax.text(4.1,2.1,'kept in the\nfitted object',fontsize=8,color=PUR,va='center')
sv(fig,'f495.png',420)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1)
json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print({k:v['min_font_printed'] for k,v in CHK.items()})

# All figures of Chapter 50. Run: python3 fig/figs.py  (PNGs into ch50/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import TargetEncoder, KBinsDiscretizer, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error as rmse
from cars import make_cars
import leak
import pandas as pd
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9,'legend.fontsize':9})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
df=make_cars(4000,0); tr,te=train_test_split(df,random_state=0)
CAT=['colour','fuel','trim','city']

# ======== 50.0-1 opener
fig,axs=plt.subplots(1,2,figsize=(5.0,1.85),dpi=240)
ax=axs[0]; s=tr.sample(500,random_state=0)
ax.scatter(s.age,s.price,s=4,color=BLUE,alpha=0.45,lw=0)
xs=np.linspace(0,20,200)[:,None]; xdf=pd.DataFrame(xs,columns=['age'])
lin=np.polyfit(tr.age,tr.price,1); ax.plot(xs[:,0],np.polyval(lin,xs[:,0]),color=GREEN,lw=1.4,ls='--',label='straight line')
kb=KBinsDiscretizer(8,encode='onehot-dense').fit(tr[['age']]); m=Ridge(1e-6).fit(kb.transform(tr[['age']]),tr.price)
ax.plot(xs[:,0],m.predict(kb.transform(xdf)),color=GREEN,lw=1.8,drawstyle='steps-post' if False else 'default',label='8 quantile bins')
ax.set_xlim(0,16); ax.set_ylim(0,48); ax.set_xlabel('age (years)'); ax.set_ylabel('price'); ax.legend(loc='upper right',handlelength=1.6); grid(ax); ax.set_title('binning bends a line',fontsize=9,color=NAVY)
NUM['open_left']={'rmse_line':float(rmse(tr.price,np.polyval(lin,tr.age))),'rmse_bins':float(rmse(tr.price,m.predict(kb.transform(tr[['age']]))))}
ax=axs[1]; vc=tr.city.value_counts(); mean_c=tr.groupby('city').price.mean(); ybar=tr.price.mean()
enc=TargetEncoder(smooth=10.0,target_type='continuous').fit(tr[['city']],tr.price); auto=dict(zip(enc.categories_[0],enc.encodings_[0]))
cs=vc.index; n_c=vc.values; raw=mean_c[cs].values-ybar; shr=np.array([auto[c] for c in cs])-ybar
for a_,b_,n_ in zip(raw,shr,n_c): ax.plot([n_,n_],[a_,b_],color=GRID,lw=0.6,zorder=1)
ax.scatter(n_c,raw,s=9,facecolor='none',edgecolor=BLUE,lw=0.7,zorder=2,label='raw mean')
ax.scatter(n_c,shr,s=8,color=GREEN,zorder=3,label='shrunk, $m=10$')
ax.axhline(0,color=GREY,lw=0.7); ax.set_xscale('log'); ax.set_xlabel('cars in the city, $n_c$'); ax.set_ylabel('encoding $-$ overall mean'); ax.set_ylim(-16,22); ax.legend(loc='upper right',handletextpad=0.1,borderaxespad=0.2); grid(ax); ax.set_title('shrinking city means',fontsize=9,color=NAVY)
NUM['open_right']={'sd_raw_small':float(raw[n_c<5].std()),'sd_shr_small':float(shr[n_c<5].std()),'sd_raw_big':float(raw[n_c>=40].std()),'sd_shr_big':float(shr[n_c>=40].std()),'n_small':int((n_c<5).sum()),'n_big':int((n_c>=40).sum())}
fig.tight_layout(w_pad=1.0); sv(fig,'f500.png',410)

# ======== 50.4-1 leakage
X,y,out=leak.run()
fig,axs=plt.subplots(1,2,figsize=(3.7,1.5),dpi=240,sharey=True)
for ax,(nm,v),tt in zip(axs,out.items(),('fit, then transform the same rows','fit_transform (cross fitted)')):
    Z=np.array(v['Z']); ax.scatter(Z,y,s=3,color=BLUE,alpha=0.35,lw=0)
    b=np.polyfit(Z,y,1); zz=np.array([Z.min(),Z.max()]); ax.plot(zz,np.polyval(b,zz),color=RED,lw=1.5)
    ax.set_title(tt,fontsize=8.5,color=NAVY); ax.set_xlabel('encoded value'); ax.text(0.04,0.95,f"corr {round(v['corr'],2)+0.0:.2f}",transform=ax.transAxes,fontsize=9,va='top',color=RED); grid(ax)
axs[0].set_ylabel('target $y$'); fig.tight_layout(w_pad=0.6); sv(fig,'f541.png',350)
NUM['leak']={k:{a:b for a,b in v.items() if a!='Z'} for k,v in out.items()}

# ======== 50.7-1 flow chart
fig,ax=plt.subplots(figsize=(3.7,1.85),dpi=240); ax.axis('off'); ax.set_xlim(-0.1,10.1); ax.set_ylim(0,5.8)
ax.text(0.0,5.55,'fit, then transform',fontsize=9,color=NAVY,weight='bold',va='center')
box(ax,0,2.95,2.5,2.1,'training\ntable',fs=8.5); box(ax,3.5,2.95,3.2,2.1,'fit: learn\ncategories_,\nbin_edges_',fs=8.5,fc='#E6F4EC',ec=GREEN); box(ax,7.5,2.95,2.5,2.1,'transform\nany rows',fs=8.5)
arr(ax,(2.55,4.0),(3.45,4.0),c=NAVY); arr(ax,(6.75,4.0),(7.45,4.0),c=NAVY)
ax.text(0.0,2.4,'TargetEncoder.fit_transform',fontsize=9,color=NAVY,weight='bold',va='center')
box(ax,0,0.35,2.5,1.5,'split rows\ninto K folds',fs=8.5); box(ax,3.5,0.35,3.2,1.5,'learn from the\nother K$-$1 folds',fs=8.5,fc='#E6F4EC',ec=GREEN); box(ax,7.5,0.35,2.5,1.5,'encode the\nheld-out fold',fs=8.5,fc=PINK,ec=RED)
arr(ax,(2.55,1.1),(3.45,1.1),c=NAVY); arr(ax,(6.75,1.1),(7.45,1.1),c=NAVY)
sv(fig,'f571.png',330)

# ======== 50.10-2 validation curves
fig,axs=plt.subplots(1,2,figsize=(4.4,1.55),dpi=240)
ax=axs[0]; Bs=[2,3,4,6,8,12,16,24,32,48]; NUM['val_bins']={}
for s_,c_,mk in (('uniform',BLUE,'o'),('quantile',GREEN,'s')):
    r_=[]
    for B in Bs:
        k=KBinsDiscretizer(B,encode='onehot-dense',strategy=s_).fit(tr[['age']]); mm=Ridge(1.0).fit(k.transform(tr[['age']]),tr.price); r_.append(rmse(te.price,mm.predict(k.transform(te[['age']]))))
    ax.plot(Bs,r_,color=c_,marker=mk,ms=3.5,lw=1.2,label=s_); NUM['val_bins'][s_]=[float(x) for x in r_]
ax.set_xscale('log'); ax.set_xticks([2,4,8,16,32]); ax.set_xticklabels(['2','4','8','16','32']); ax.minorticks_off(); ax.set_xlabel('number of bins $B$'); ax.set_ylabel('test RMSE'); ax.legend(loc='upper right'); grid(ax)
ax=axs[1]; CATC=CAT; sm=[0.01,1,10,100,1000,1e4]; r_=[]
def tpipe(s):
    ct=ColumnTransformer([('cat',TargetEncoder(smooth=s,target_type='continuous',cv=KFold(5,shuffle=True,random_state=0)),CATC),('num','passthrough',['age','km'])]); return make_pipeline(ct,HGB(random_state=0))
for s_ in sm: r_.append(rmse(te.price,tpipe(s_).fit(tr[CAT+['age','km']],tr.price).predict(te[CAT+['age','km']])))
ra=rmse(te.price,tpipe('auto').fit(tr[CAT+['age','km']],tr.price).predict(te[CAT+['age','km']]))
ax.plot(sm,r_,color=PUR,marker='o',ms=3.5,lw=1.2); ax.axhline(ra,color=GREEN,lw=1.3,ls='--'); ax.text(0.02,ra,'"auto"',color=GREEN,fontsize=9,transform=ax.get_yaxis_transform(),va='bottom')
ax.set_xscale('log'); ax.set_xticks([0.01,1,100,1e4]); ax.set_xticklabels(['0.01','1','100','10 000']); ax.minorticks_off(); ax.set_xlabel('smoothing $m$'); grid(ax); ax.set_ylabel('test RMSE')
NUM['val_smooth']={'m':sm,'rmse':[float(x) for x in r_],'auto':float(ra)}
fig.tight_layout(w_pad=0.8); sv(fig,'f5102.png',340)

# ======== 50.11-1 cardinality
R=json.load(open('verify/cardinality.json')); fig,ax=plt.subplots(figsize=(4.4,1.7),dpi=240)
sty={'ridge one-hot':(BLUE,'o','-'),'ridge target':(BLUE,'^','-'),'hgb one-hot':(OR,'o','--'),'hgb ordinal':(OR,'s','--'),'hgb target':(OR,'^','--')}
for k,v in R['rmse'].items():
    c_,mk,ls=sty[k]; ax.plot(R['cities'],v,color=c_,marker=mk,ms=4,lw=1.3,ls=ls,label=k.replace('hgb','HGB').replace('ridge','ridge'))
ax.set_xscale('log'); ax.set_xticks(R['cities']); ax.set_xticklabels([str(c) for c in R['cities']]); ax.minorticks_off(); ax.set_xlabel('number of cities in the table'); ax.set_ylabel('test RMSE'); ax.set_ylim(1.0,4.6)
ax.legend(loc='upper left',ncol=2,columnspacing=1.0,handlelength=2.2,borderaxespad=0.2); grid(ax)
fig.tight_layout(); sv(fig,'f5111.png',340)
json.dump({'num':NUM,'font_check':CHK},open('verify/fig_numbers.json','w'),indent=1)
for k,v in CHK.items(): print(k,v)

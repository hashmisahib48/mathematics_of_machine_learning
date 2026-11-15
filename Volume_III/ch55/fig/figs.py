# All figures of Chapter 55. Run: python3 fig/figs.py  (PNGs into ch55/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.model_selection import (KFold, StratifiedKFold, ShuffleSplit, TimeSeriesSplit, GroupKFold,
    StratifiedGroupKFold, LeaveOneGroupOut, GroupShuffleSplit)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from cvdata import make_patients, make_sales
plt.rcParams.update({'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.labelsize':8,'legend.fontsize':7.8})
NUM={}; CHK={}
J=lambda f: json.load(open('verify/'+f))
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ======== 55.0-1 opener: the estimate against the truth, two stories
cg=J('case_groups.json'); cs=J('case_series.json')
fig,axs=plt.subplots(1,2,figsize=(3.75,1.85),dpi=240,gridspec_kw={'width_ratios':[0.8,1.2]})
ax=axs[0]; labs=['KFold','Group\nKFold','new\npatients']; vals=[cg['KFold, shuffled'][0],cg['GroupKFold'][0],cg['truth'][0]]
bars=ax.bar(range(3),vals,color=[OR,GREEN,'#B8BEC8'],width=0.62,edgecolor=NAVY,lw=0.6)
for b_,h in zip(bars,['//','','']): b_.set_hatch(h)
for k,v in enumerate(vals): ax.text(k,v+0.015,'%.2f'%v,ha='center',fontsize=7.8,color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=7.8); ax.set_ylim(0,1.05); ax.set_ylabel('accuracy'); ax.set_title('patients, 8 visits each',fontsize=8,color=NAVY); grid(ax)
ax=axs[1]; w=0.26; xs=np.arange(2)
for j,(c_,col,h,lab) in enumerate((('KFold',OR,'//','KFold'),('TimeSeriesSplit',GREEN,'','TimeSeriesSplit'),('next 60 weeks','#B8BEC8','','next 60 weeks'))):
    v=[cs['random forest|'+c_][0],cs['ridge|'+c_][0]]
    ax.bar(xs+(j-1)*w,v,w*0.92,color=col,hatch=h,edgecolor=NAVY,lw=0.6,label=lab)
    for x_,v_ in zip(xs+(j-1)*w,v): ax.text(x_,v_+0.3,'%.1f'%v_,ha='center',fontsize=7.8,color=NAVY)
ax.set_xticks(xs); ax.set_xticklabels(['forest','ridge'],fontsize=7.8); ax.set_ylim(0,34); ax.set_ylabel('RMSE (lower is better)'); ax.set_title('weekly sales',fontsize=8,color=NAVY)
ax.legend(loc='upper left',fontsize=7.6,handlelength=1.2,labelspacing=0.15,borderaxespad=0.0,ncol=1,framealpha=1,edgecolor=GRID); grid(ax)
fig.tight_layout(w_pad=0.8); sv(fig,'f550.png',360)

# ======== 55.2-1 the index plot of eight splitters
gs_=np.array([6,5,4,4,3,2]); g=np.repeat(np.arange(6),gs_); cls_g=np.array([0,1,0,0,1,0]); y=cls_g[g]; n=len(g)
X=np.zeros((n,1))
panels=[('KFold(4, shuffle=True)',KFold(4,shuffle=True,random_state=0),{}),
        ('StratifiedKFold(4, shuffle=True)',StratifiedKFold(4,shuffle=True,random_state=0),{}),
        ('TimeSeriesSplit(4)',TimeSeriesSplit(4),{}),
        ('GroupKFold(3)',GroupKFold(3),{}),
        ('StratifiedGroupKFold(3, shuffle=True)',StratifiedGroupKFold(3,shuffle=True,random_state=0),{}),
        ('LeaveOneGroupOut()',LeaveOneGroupOut(),{})]
cm=ListedColormap(['#FFFFFF',BLUE,OR])
fig=plt.figure(figsize=(4.9,2.7),dpi=240); gsp=fig.add_gridspec(4,2,height_ratios=[0.55,1,1,1.4],hspace=0.5,wspace=0.12,left=0.02,right=0.995,top=0.995,bottom=0.01)
ax=fig.add_subplot(gsp[0,:]); ax.imshow(np.vstack([y,np.zeros(n)]),cmap=ListedColormap([LGREY,NAVY]),aspect='auto',vmin=0,vmax=1,extent=(0,n,0,2))
ax.imshow(np.vstack([np.zeros(n)]),alpha=0)
ax.cla()
for i in range(n):
    ax.add_patch(Rectangle((i,1),1,1,fc=NAVY if y[i] else LGREY,ec='white',lw=0.6)); ax.add_patch(Rectangle((i,0),1,1,fc=('#F2F4F8' if g[i]%2==0 else '#D9DEE7'),ec='white',lw=0.6))
    ax.text(i+0.5,0.5,str(g[i]+1),ha='center',va='center',fontsize=7.8,color=NAVY)
ax.set_xlim(0,n); ax.set_ylim(0,2); ax.axis('off'); ax.text(-0.5,1.5,'class 1 dark',ha='right',va='center',fontsize=7.8,color=NAVY); ax.text(-0.5,0.5,'group',ha='right',va='center',fontsize=7.8,color=NAVY)
ax.set_title('the 24 samples: class (top) and group (bottom, number = patient)',fontsize=7.8,color=NAVY,pad=3)
folds={}
for k,(ttl,cv,kw) in enumerate(panels):
    ax=fig.add_subplot(gsp[1+k//2,k%2]); rows=[]; 
    for tr,te in cv.split(X,y,g if 'Group' in ttl else None):
        r_=np.zeros(n); r_[tr]=1; r_[te]=2; rows.append(r_)
    rows=np.array(rows); folds[ttl]=rows.shape[0]
    ax.imshow(rows,cmap=cm,vmin=0,vmax=2,aspect='auto',interpolation='nearest')
    ax.set_xticks(np.arange(-0.5,n,1),minor=True); ax.set_yticks(np.arange(-0.5,rows.shape[0],1),minor=True); ax.grid(which='minor',color='white',lw=0.7); ax.tick_params(which='both',length=0)
    ax.set_xticks([]); ax.set_yticks(range(rows.shape[0])); ax.set_yticklabels([str(i+1) for i in range(rows.shape[0])],fontsize=7.8)
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.set_title(ttl,fontsize=7.8,color=NAVY,pad=2)
NUM['index_plot']=folds
fig.text(0.985,0.003,'',fontsize=7.8)
sv(fig,'f552.png',470)

# ======== 55.5-1 the algorithm: what cross_validate does
fig,ax=plt.subplots(figsize=(3.55,1.5),dpi=240); ax.set_xlim(-0.15,10.1); ax.set_ylim(0,3.6); ax.axis('off')
box(ax,0.0,2.3,1.9,1.0,'data\n$X,\\,y,\\,$groups',fc=LB,ec=BLUE,fs=9)
box(ax,2.5,2.3,2.0,1.0,'cv.split\n(X, y, groups)',fc='#EFE6F5',ec=PUR,fs=9)
box(ax,5.1,2.3,2.0,1.0,'train, test\nindex pair',fc=LGREY,ec=GREY,fs=9)
box(ax,7.7,2.3,2.2,1.0,'clone\n(estimator)',fc='#E3F2E9',ec=GREEN,fs=9)
box(ax,7.7,0.5,2.2,1.0,'fit on train',fc='#E3F2E9',ec=GREEN,fs=9)
box(ax,5.1,0.5,2.0,1.0,'score on test\n$s_k$',fc=PINK,ec=RED,fs=9)
box(ax,2.5,0.5,2.0,1.0,'repeat for\neach pair',fc=LGREY,ec=GREY,fs=9)
box(ax,0.0,0.5,1.9,1.0,'scores\nmean, sd',fc=LB,ec=BLUE,fs=9)
for p,q in (((1.9,2.8),(2.5,2.8)),((4.5,2.8),(5.1,2.8)),((7.1,2.8),(7.7,2.8)),((8.8,2.3),(8.8,1.5)),((7.7,1.0),(7.1,1.0)),((5.1,1.0),(4.5,1.0)),((2.5,1.0),(1.9,1.0))): arr(ax,p,q,c=NAVY,lw=1.0,ms=7)
ax.text(3.5,1.95,'K pairs',ha='center',fontsize=7.8,color=PUR)
sv(fig,'f555.png',340)

# ======== 55.8-1 behaviour: bias, spread, winner's curse
kb=J('kbias.json'); pn=J('prose_numbers.json'); ne=J('nested.json')
fig,axs=plt.subplots(1,3,figsize=(4.6,1.95),dpi=240,gridspec_kw={'width_ratios':[1,1,1.05]})
ax=axs[0]; Ks=kb['Ks']; ax.axhline(0,color=GREY,lw=0.8,ls=':'); ax.plot(range(len(Ks)),[100*b_ for b_ in kb['bias']],'-o',color=PUR,ms=4,lw=1.3)
ax.set_xticks(range(len(Ks))); ax.set_xticklabels(Ks); ax.set_xlabel('number of folds $K$'); ax.set_ylabel('bias (points)'); ax.set_title('(a) pessimism',fontsize=8,color=NAVY); ax.set_ylim(-1.6,0.6); grid(ax)
ax=axs[1]; vals=[100*pn['naive_se'],100*kb['sd_partition_5fold'],100*pn['sd_true']]; labs=['naive\ns.e.','new\npartition','new\ndata']
ax.bar(range(3),vals,color=[RED,OR,BLUE],width=0.62,edgecolor=NAVY,lw=0.6,hatch=None)
for k,v in enumerate(vals): ax.text(k,v+0.15,'%.1f'%v,ha='center',fontsize=7.8,color=NAVY)
ax.set_xticks(range(3)); ax.set_xticklabels(labs,fontsize=7.8); ax.set_ylim(0,9.4); ax.set_ylabel('spread (points)'); ax.set_title('(b) 5-fold spread',fontsize=8,color=NAVY); grid(ax)
ax=axs[2]; rngj=np.random.RandomState(0)
for k,(key,col,mk) in enumerate((('best_cv_values',OR,'s'),('nested_values',GREEN,'o'),('fresh_values','#7F8793','^'))):
    v=np.array(ne[key])[:120]; ax.scatter(k+rngj.uniform(-0.28,0.28,len(v)),v,s=5,color=col,marker=mk,lw=0,alpha=0.7)
    ax.plot([k-0.34,k+0.34],[np.mean(ne[key])]*2,color=NAVY,lw=1.4)
ax.axhline(0.5,color=PUR,lw=0.9,ls=':'); ax.set_xticks(range(3)); ax.set_xticklabels(['best\nof 20','nested','fresh\ndata'],fontsize=7.8); ax.set_xlim(-0.6,2.6); ax.set_ylabel('accuracy'); ax.set_title('(c) winner’s curse',fontsize=8,color=NAVY); grid(ax)
fig.tight_layout(w_pad=0.7); sv(fig,'f556.png',440)

# ======== 55.9-1 the sales series, the five folds and two forecasts
X,yv=make_sales(300,0); t=np.arange(300); ts=TimeSeriesSplit(5,test_size=30)
fig,ax=plt.subplots(figsize=(4.4,1.95),dpi=240)
ax.plot(t,yv,color='#9AA3AF',lw=0.8,zorder=1,label='sales')
for k,(tr,te) in enumerate(ts.split(X[:240])):
    ax.axvspan(te[0]-0.5,te[-1]+0.5,color=OR if k%2==0 else '#F7C873',alpha=0.55,zorder=0,lw=0)
    ax.text(te.mean(),yv.min()-4.6,str(k+1),ha='center',fontsize=7.8,color=NAVY)
ax.axvspan(239.5,299.5,color=LGREY,zorder=0,lw=0)
rf=RandomForestRegressor(50,random_state=0).fit(X[:240],yv[:240]); rd=Ridge(alpha=1.0).fit(X[:240],yv[:240])
ax.plot(t[240:],rf.predict(X[240:]),color=RED,lw=1.4,ls='--',label='forest forecast'); ax.plot(t[240:],rd.predict(X[240:]),color=GREEN,lw=1.4,label='ridge forecast')
ax.axvline(239.5,color=NAVY,lw=0.8); ax.text(270,yv.min()-4.6,'next 60 weeks',ha='center',fontsize=7.8,color=NAVY)
ax.set_xlim(0,300); ax.set_ylim(yv.min()-7,yv.max()+4); ax.set_xlabel('week'); ax.set_ylabel('sales'); ax.legend(loc='upper left',fontsize=7.8,handlelength=1.5,labelspacing=0.2,borderaxespad=0.2,framealpha=1,edgecolor=GRID); grid(ax)
NUM['series_range']=[float(yv.min()),float(yv.max())]
fig.tight_layout(); sv(fig,'f557.png',420)

json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
print(json.dumps(CHK,indent=0)); print(NUM)

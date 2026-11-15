# All figures of Chapter 8.  Run: python3 fig/figs.py  (PNGs go into ch08/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from matplotlib.patches import Rectangle, Circle
from sklearn.datasets import load_diabetes, load_digits, load_iris
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; FC={}
def save2(fig,name,wpx):
    """save and record smallest printed font size (design size x printed width / image width)"""
    fs=[t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()]
    save(fig,name); from PIL import Image
    iw=Image.open(name).width/240.0; sc=(wpx/96.0)/iw; FC[name]=[round(min(fs),1),round(sc,3),round(min(fs)*sc,2)]
    if min(fs)*sc<7.5: print('SMALL FONT',name,FC[name])
LGB='#DCE6F5'; GRY='#9AA3AF'; LGR='#D5D9E0'
def cell(ax,x,y,w,h,fc,ec='white',lw=0.8,txt=None,fs=8.5,tc=NAVY,**k):
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,**k))
    if txt is not None: ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc)

# ---------- 8.0-1 opener: the data budget
fig,ax=plt.subplots(figsize=(5.4,2.55)); ax.axis('off'); ax.set_xlim(-2.2,10.6); ax.set_ylim(-0.2,5.2)
cell(ax,0,4.2,10,0.7,LGB,ec=BLUE,txt='all $n$ labelled samples',fs=9.5)
ax.text(-0.15,4.55,'data',ha='right',va='center',fontsize=9.5,color=NAVY,weight='bold')
cell(ax,0,2.75,7.5,0.7,BLUE,txt='training part',fs=9.5,tc='white'); cell(ax,7.5,2.75,2.5,0.7,GRY,txt='test part',fs=9.5,tc='white')
ax.text(-0.15,3.1,'step 1',ha='right',va='center',fontsize=9.5,color=NAVY,weight='bold')
ax.text(5.35,3.82,'split once',ha='left',va='center',fontsize=9,color=GREY,style='italic'); arr(ax,(5,4.15),(5,3.5),c=GREY,lw=0.9)
for j in range(5):
    cell(ax,1.5*j,1.3,1.5,0.7,OR if j==3 else '#7FA6DE',ec='white',lw=1.5,txt='held out' if j==3 else 'train',fs=8.5,tc='white' if j!=3 else NAVY)
ax.text(-0.15,1.65,'step 2',ha='right',va='center',fontsize=9.5,color=NAVY,weight='bold')
ax.text(1.1,2.42,'cross-validate: five folds in turn',ha='left',va='center',fontsize=9,color=GREY,style='italic'); arr(ax,(0.75,2.7),(0.75,2.05),c=GREY,lw=0.9)
cell(ax,7.5,0.0,2.5,0.7,GRY,txt='test part',fs=9.5,tc='white'); ax.text(-0.15,0.35,'step 3',ha='right',va='center',fontsize=9.5,color=NAVY,weight='bold')
ax.text(3.75,0.35,'pick the best model; refit it on the training part',ha='center',va='center',fontsize=9,color=BLUE)
ax.text(8.15,1.75,'used once,\nat the end',ha='left',va='center',fontsize=9,color=GREY,style='italic'); arr(ax,(7.9,2.7),(7.9,0.78),c=GREY,lw=0.9,ls=(0,(3,2)))
save2(fig,'f800.png',430)

# ---------- 8.1-1 how train_test_split deals the samples
rng=np.random.default_rng(9); n20=20; y20=np.zeros(n20,int); y20[rng.choice(n20,6,replace=False)]=1; idx=np.arange(n20)
NUM['split20']={'n_pos':int(y20.sum())}
fig,ax=plt.subplots(figsize=(5.4,2.85)); ax.axis("off"); ax.set_xlim(-0.2,21.2); ax.set_ylim(1.5,12.0)
def row(ax,ids,x0,yy,wd=1.05):
    for k,i in enumerate(ids):
        cell(ax,x0+k*wd,yy,wd*0.92,0.9,OR if y20[i]==1 else BLUE,txt=str(i),fs=8.5,tc=NAVY if y20[i]==1 else 'white')
ax.text(0,11.5,'the 20 samples in their original order  (orange = class 1, six of them)',fontsize=9,color=NAVY,va='center'); row(ax,idx,0,10.0)
for lab,strat,yy in [('plain split:  train_test_split(X, y, test_size=0.25, random_state=0)',None,6.6),('with stratify=y',y20,2.4)]:
    a,b_,_,_=train_test_split(idx,y20,test_size=0.25,random_state=0,stratify=strat)
    ax.text(0,yy+2.0,lab,fontsize=9,color=PUR,va='center',weight='bold')
    row(ax,a,0,yy); row(ax,b_,15.9,yy)
    ax.add_patch(Rectangle((15.75,yy-0.12),5.4,1.14,fc='none',ec=GREY,lw=1.4,ls=(0,(3,2))))
    ax.text(0,yy-0.55,'training part: 15 samples, %d in class 1'%y20[a].sum(),fontsize=9,color=NAVY,va='center'); ax.text(21.15,yy-0.55,'test part: %d of 5 in class 1'%y20[b_].sum(),fontsize=9,color=GREY,va='center',ha='right')
    NUM['split20']['plain' if strat is None else 'strat']={'test_idx':[int(v) for v in b_],'test_pos':int(y20[b_].sum()),'train_pos':int(y20[a].sum())}
save2(fig,'f811.png',440)

# ---------- 8.2-1 baselines: regression and classification
d=load_diabetes(); Xd_tr,Xd_te,yd_tr,yd_te=train_test_split(d.data,d.target,test_size=0.25,random_state=0)
fig,axs=plt.subplots(1,2,figsize=(5.4,2.45),gridspec_kw={'width_ratios':[1.05,1]})
ax=axs[0]; ax.hist(yd_tr,bins=18,color=LGB,ec='white'); mu,md=yd_tr.mean(),np.median(yd_tr)
ax.axvline(mu,color=OR,lw=1.8); ax.axvline(md,color=PUR,lw=1.8,ls=(0,(4,2)))
ax.text(mu+6,52,'mean\n%.1f'%mu,color=OR,fontsize=9,va='top',ha='left',weight='bold'); ax.text(md-6,52,'median\n%.0f'%md,color=PUR,fontsize=9,va='top',ha='right',weight='bold')
ax.set_xlabel('diabetes target (training part)'); ax.set_ylabel('number of patients'); ax.set_title('Two constant predictors',fontsize=9.5,color=NAVY); ax.set_ylim(0,54)
Xg,yg=load_digits(return_X_y=True); yg=(yg==9).astype(int); Xg_tr,Xg_te,yg_tr,yg_te=train_test_split(Xg,yg,test_size=0.25,random_state=0,stratify=yg)
names=['most_frequent','prior','stratified','uniform']; acc=[DummyClassifier(strategy=s,random_state=0).fit(Xg_tr,yg_tr).score(Xg_te,yg_te) for s in names]
NUM['dummy_acc']={n:round(float(a),3) for n,a in zip(names,acc)}
ax=axs[1]; cols=[BLUE,BLUE,'#7FA6DE','#7FA6DE']; ax.barh(range(4),acc,color=cols,height=0.62,hatch=None)
for k,a in enumerate(acc): ax.text(a+0.02,k,f'{a:.3f}',va='center',fontsize=9)
ax.set_yticks(range(4)); ax.set_yticklabels(['most_frequent','prior','stratified','uniform'],fontsize=8.5); ax.invert_yaxis(); ax.set_xlim(0,1.2); ax.set_xlabel('accuracy'); ax.set_title('DummyClassifier, nine or not',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.2); save2(fig,'f821.png',450)

# ---------- 8.3-1 regression scores on five predictions, and R2 on new data
yv=np.array([3.,5,8,10,14]); yp=np.array([4.,5,7,12,12]); rr=yv-yp; ii=np.arange(1,6)
NUM['reg5']={'mse':float((rr**2).mean()),'rmse':float(np.sqrt((rr**2).mean())),'mae':float(np.abs(rr).mean()),'r2':float(1-(rr**2).sum()/((yv-yv.mean())**2).sum())}
xt=np.array([1.,2,3,4,5]); yt=np.array([3.,4,6,7,10]); xn=np.array([6.,7,8,9]); yn=np.array([11.,13,12,14]); m=LinearRegression().fit(xt[:,None],yt); pn=m.predict(xn[:,None])
sse=float(((yn-pn)**2).sum()); sst=float(((yn-yn.mean())**2).sum()); NUM['negr2']={'train_r2':float(m.score(xt[:,None],yt)),'sse':sse,'sst':sst,'test_r2':1-sse/sst,'mean_new':float(yn.mean())}
fig,axs=plt.subplots(1,2,figsize=(5.5,2.55),gridspec_kw={'width_ratios':[1.0,1.15]})
ax=axs[0]; ax.axhline(yv.mean(),color=OR,lw=1.2,ls='--'); ax.text(5.45,yv.mean()-0.5,'mean  $\\bar y$ = 8',color=OR,fontsize=8.8,ha='right',va='top')
for a_,u,v in zip(ii,yv,yp): ax.plot([a_,a_],[u,v],color=RED,lw=2.2,zorder=3)
ax.scatter(ii,yv,s=40,color=BLUE,zorder=5,edgecolor='white',lw=0.8,label='measured $y$'); ax.scatter(ii,yp,s=40,marker='D',facecolor='white',edgecolor=GREEN,lw=1.5,zorder=6,label='predicted $\\hat y$')
ax.set_xlabel('sample $i$'); ax.set_ylabel('target'); ax.set_xlim(0.5,5.5); ax.set_ylim(1,17); ax.legend(loc='upper left',fontsize=8.5,handletextpad=0.2,borderaxespad=0.2); ax.set_xticks(ii); ax.grid(color=GRID,lw=0.5); ax.set_title('(a) Example 8.3-1',fontsize=9.5,color=NAVY)
ax=axs[1]; xx=np.array([0,9.6]); ax.plot(xx,m.predict(xx[:,None]),color=GREEN,lw=1.8)
ax.plot([5.0,9.6],[yn.mean()]*2,color=OR,lw=1.4,ls='--'); ax.text(4.9,yn.mean()+0.3,'mean of new data',color=OR,fontsize=8.8,ha='right',va='bottom')
for a_,u,v in zip(xn,yn,pn): ax.plot([a_,a_],[u,v],color=RED,lw=1.8,zorder=3)
ax.scatter(xt,yt,s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5,label='training data'); ax.scatter(xn,yn,s=40,color=GREY,marker='s',edgecolor='white',lw=0.8,zorder=5,label='new data')
ax.text(0.2,19.0,f'training $R^2$ = {NUM["negr2"]["train_r2"]:.3f}',fontsize=9,color=BLUE,va='top'); ax.text(0.2,17.2,'$R^2$ on new data = \u2212%.2f'%(-NUM['negr2']['test_r2']),fontsize=9,color=RED,va='top',weight='bold')
ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); ax.set_xlim(0,9.7); ax.set_ylim(0,19.4); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(loc='lower right',fontsize=8.5,handletextpad=0.2,borderaxespad=0.2)
ax.set_title('(b) a line on longer rides',fontsize=9.5,color=NAVY)
fig.tight_layout(w_pad=1.0); save2(fig,'f831.png',470)

# ---------- 8.4-1 ten samples and their confusion matrix
yt10=np.array([1,1,1,1,0,0,0,0,0,0]); yp10=np.array([1,1,1,0,1,1,0,0,0,0]); cm=confusion_matrix(yt10,yp10); NUM['cm10']=cm.tolist()
fig,axs=plt.subplots(1,2,figsize=(5.4,2.7),gridspec_kw={'width_ratios':[1.0,1.05]})
ax=axs[0]; ax.axis('off'); ax.set_xlim(-2.3,10.4); ax.set_ylim(-1.4,4.9); ax.set_aspect('equal')
ax.text(-0.2,3.3,'true',ha='right',va='center',fontsize=9,color=NAVY); ax.text(-0.2,1.5,'predicted',ha='right',va='center',fontsize=9,color=NAVY)
for k in range(10):
    for row_,v in ((3.3,yt10[k]),(1.5,yp10[k])):
        ax.add_patch(Circle((k+0.5,row_),0.42,fc=OR if v==1 else BLUE,ec='white',lw=0.8)); ax.text(k+0.5,row_,str(v),ha='center',va='center',fontsize=8.5,color=NAVY if v==1 else 'white')
    if yt10[k]!=yp10[k]: ax.plot([k+0.5,k+0.5],[2.0,2.8],color=RED,lw=1.6); ax.add_patch(Circle((k+0.5,1.5),0.52,fc='none',ec=RED,lw=1.4)); ax.text(k+0.5,0.5,'FN' if yt10[k]==1 else 'FP',ha='center',va='center',fontsize=8.5,color=RED,weight='bold')
ax.text(4.2,4.45,'ten samples; class 1 (orange) is sought',ha='center',va='center',fontsize=9,color=GREY)
ax.text(4.2,-1.05,'red: wrong prediction',ha='center',va='center',fontsize=9,color=RED)
ax=axs[1]; ax.axis('off'); ax.set_xlim(-2.2,4.3); ax.set_ylim(-0.9,4.3)
quad=[(0,1,'TN',cm[0,0],LGB,BLUE),(1,1,'FP',cm[0,1],PINK,RED),(0,0,'FN',cm[1,0],PINK,RED),(1,0,'TP',cm[1,1],'#E3F4EA',GREEN)]
for cx,cy,nm,v,fc,ec in quad:
    ax.add_patch(Rectangle((cx*1.5,cy*1.2+0.3),1.45,1.15,fc=fc,ec=ec,lw=1.3)); ax.text(cx*1.5+0.72,cy*1.2+1.1,str(v),ha='center',va='center',fontsize=15,color=NAVY,weight='bold'); ax.text(cx*1.5+0.72,cy*1.2+0.62,nm,ha='center',va='center',fontsize=9,color=ec,weight='bold')
ax.text(0.72,3.0,'pred. 0',ha='center',fontsize=9,color=NAVY); ax.text(2.22,3.0,'pred. 1',ha='center',fontsize=9,color=NAVY)
ax.text(-0.1,2.5,'true 0',ha='right',va='center',fontsize=9,color=NAVY); ax.text(-0.1,1.3,'true 1',ha='right',va='center',fontsize=9,color=NAVY)
ax.text(1.5,3.55,'confusion matrix',ha='center',fontsize=9.5,color=NAVY,weight='bold')
ax.text(1.5,-0.45,'precision = TP/(TP+FP) = 3/5\nrecall = TP/(TP+FN) = 3/4',ha='center',va='center',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=0.5); save2(fig,'f841.png',450)

# ---------- 8.5-1 five-fold cross-validation on 15 samples
n15=15; ids=np.arange(n15); fig,axs=plt.subplots(2,1,figsize=(5.4,3.9),gridspec_kw={'height_ratios':[1.6,1]})
ax=axs[0]; ax.axis('off'); ax.set_xlim(-2.6,17.6); ax.set_ylim(-0.5,5.9)
ax.text(7.5,5.55,'KFold(n_splits=5): each fold is the held-out part once',ha='center',fontsize=9,color=NAVY,weight='bold')
for k,(tr,te) in enumerate(KFold(5).split(ids)):
    yy=4.5-k*0.95
    for i in ids: cell(ax,i,yy,0.92,0.78,OR if i in te else '#7FA6DE',txt=None)
    ax.text(-0.2,yy+0.39,f'split {k+1}',ha='right',va='center',fontsize=9,color=NAVY)
    ax.text(15.3,yy+0.39,f'$s_{k+1}$',ha='left',va='center',fontsize=9.5,color=OR if False else NAVY)
    ax.annotate('',xy=(16.7,yy+0.39),xytext=(16.0,yy+0.39),arrowprops=dict(arrowstyle='-|>',color=GREY,lw=0.8))
ax.text(7.5,-0.05,'blue: samples used to fit      orange: held-out fold, scored as $s_j$',ha='center',va='center',fontsize=9,color=GREY)
ax=axs[1]; ax.axis('off'); ax.set_xlim(-2.6,17.6); ax.set_ylim(-0.2,3.4)
fold_ord=np.zeros(n15,int)
for k,(tr,te) in enumerate(KFold(5).split(ids)): fold_ord[te]=k+1
fold_shuf=np.zeros(n15,int)
for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=0).split(ids)): fold_shuf[te]=k+1
NUM['fold_shuf']=fold_shuf.tolist()
fcol=['#7FA6DE',OR,TEAL,PUR,GRY]
for yy,fo,lab in ((1.8,fold_ord,'shuffle=False'),(0.5,fold_shuf,'shuffle=True')):
    for i in ids: cell(ax,i,yy,0.92,0.78,fcol[fo[i]-1],txt=str(fo[i]),fs=8.5,tc='white' if fo[i] in (3,4) else NAVY)
    ax.text(-0.2,yy+0.39,lab,ha='right',va='center',fontsize=8.8,color=NAVY)
ax.text(7.5,3.0,'which fold each sample belongs to',ha='center',fontsize=9,color=NAVY,weight='bold')
fig.tight_layout(h_pad=0.3); save2(fig,'f851.png',440)

# ---------- 8.5-2 iris: what is in each test fold
X_i,y_i=load_iris(return_X_y=True)
fig,axs=plt.subplots(1,3,figsize=(5.4,2.3),sharey=True); ccol=[BLUE,OR,TEAL]; NUM['iris_folds']={}
for ax,(t,cv) in zip(axs,[('KFold(3)',KFold(3)),('KFold(3), shuffled',KFold(3,shuffle=True,random_state=0)),('StratifiedKFold(3)',StratifiedKFold(3))]):
    comp=np.array([[np.sum(y_i[te]==c_) for c_ in range(3)] for _,te in cv.split(X_i,y_i)]); NUM['iris_folds'][t]=comp.tolist()
    bot=np.zeros(3)
    for c_ in range(3): ax.bar(range(1,4),comp[:,c_],bottom=bot,color=ccol[c_],width=0.7,ec='white',hatch=['','//','..'][c_]); bot+=comp[:,c_]
    ax.set_title(t,fontsize=8.8,color=NAVY); ax.set_xticks([1,2,3]); ax.set_xlabel('test fold')
axs[0].set_ylabel("samples in the test fold"); axs[0].set_ylim(0,56)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc=ccol[c_],hatch=['','//','..'][c_],ec='white',label=['setosa','versicolor','virginica'][c_]) for c_ in range(3)],loc='lower center',ncol=3,fontsize=8.5,columnspacing=1.2,handlelength=1.4,handletextpad=0.4)
fig.tight_layout(w_pad=0.6,rect=[0,0.09,1,1]); save2(fig,'f852.png',450)

# ---------- 8.5-3 mean and spread of fold scores
Xd,yd=d.data,d.target; ols=LinearRegression(); cfgs=[('no shuffle',KFold(5))]+[(f'shuffle, seed {s_}',KFold(5,shuffle=True,random_state=s_)) for s_ in range(3)]
fig,ax=plt.subplots(figsize=(5.0,2.55)); NUM['cvspread']={}
for k,(nm,cv) in enumerate(cfgs):
    sc_=cross_val_score(ols,Xd,yd,cv=cv); sb=cross_val_score(DummyRegressor(),Xd,yd,cv=cv)
    NUM['cvspread'][nm]={'folds':[round(float(v),3) for v in sc_],'mean':round(float(sc_.mean()),3),'sd':round(float(sc_.std(ddof=1)),3),'base_mean':round(float(sb.mean()),3)}
    xs=k+np.linspace(-0.22,0.22,5); ax.scatter(xs,sc_,s=26,color=BLUE,zorder=4,edgecolor='white',lw=0.6,label='least squares' if k==0 else None)
    ax.scatter(xs,sb,s=26,color=GRY,marker='s',zorder=4,edgecolor='white',lw=0.6,label='mean baseline' if k==0 else None)
    ax.plot([k-0.3,k+0.3],[sc_.mean()]*2,color=GREEN,lw=2.2,zorder=5,label='mean of the five (least squares)' if k==0 else None)
    ax.text(k,0.66,f'{sc_.mean():.3f}',ha='center',fontsize=9,color=GREEN,weight='bold')
ax.axhline(0.359,color=RED,lw=1.1,ls=(0,(4,2))); ax.text(3.45,0.335,'single test split: 0.359',color=RED,fontsize=8.8,ha='right',va='top')
ax.set_xticks(range(4)); ax.set_xticklabels([c_[0] for c_ in cfgs],fontsize=8.8); ax.set_ylabel('$R^2$ on the held-out fold'); ax.set_ylim(-0.2,0.75); ax.set_xlim(-0.5,3.5); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
h_,l_=ax.get_legend_handles_labels(); fig.legend(h_,l_,loc='lower center',ncol=3,fontsize=8.5,columnspacing=0.9,handletextpad=0.2)
fig.tight_layout(rect=[0,0.08,1,1]); save2(fig,'f853.png',410)

# ---------- 8.6-1 error and score, same ranking
cands={'mean baseline':[2],'bmi only':[2],'bmi, bp, s5':[2,3,8],'all ten':list(range(10))}; mse={}
for nm,cols in cands.items():
    m_=DummyRegressor() if nm=='mean baseline' else LinearRegression()
    mse[nm]=float(-cross_val_score(m_,Xd[:,cols],yd,cv=5,scoring='neg_mean_squared_error').mean())
NUM['mse_cv']={k:round(v,1) for k,v in mse.items()}
fig,axs=plt.subplots(1,2,figsize=(4.5,2.1),sharey=True); nm_=list(mse)
axs[0].barh(range(4),[mse[k] for k in nm_],color=RED,height=0.6); axs[1].barh(range(4),[-mse[k] for k in nm_],color=PUR,height=0.6)
for k,v in enumerate(mse.values()): axs[0].text(v+60,k,f'{v:.0f}',va='center',fontsize=8.8); axs[1].text(-v-60,k,f'−{v:.0f}',va='center',ha='right',fontsize=8.8)
axs[0].set_yticks(range(4)); axs[0].set_yticklabels(nm_,fontsize=8.8); axs[0].invert_yaxis(); axs[0].set_xlim(0,7800); axs[1].set_xlim(-7800,0)
axs[0].set_title('error (MSE): lower is better',fontsize=9,color=NAVY); axs[1].set_title('scorer value: higher is better',fontsize=9,color=NAVY)
axs[0].set_xlabel('cross-validated MSE'); axs[1].set_xlabel('neg_mean_squared_error'); axs[1].spines['right'].set_visible(True); axs[1].spines['left'].set_visible(False); axs[1].tick_params(left=False)
fig.tight_layout(w_pad=0.6); save2(fig,'f861.png',400)

# ---------- 8.7-1 the protocol as a flow chart
fig,ax=new(4.6,5.7)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.8,tc=NAVY,bold=False): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,tc=tc,bold=bold,r=0.08)
A=lambda p,q,c=NAVY,**k: arr(ax,p,q,c=c,lw=0.9,ms=8,**k)
XL=3.3; XR=9.3; H2=1.05
fb(5.2,12.2,4.6,0.7,'all labelled data',fc=LGREY,ec=GREY)
fb(5.2,10.8,7.4,0.85,'1  Split once:  train_test_split(...)',fc='#F1EAF7',ec=PUR)
fb(XL,9.3,5.8,0.7,'training part',fc=BLUE,ec=BLUE,tc='white',bold=True); fb(XR,9.3,2.6,0.7,'test part',fc=GRY,ec=GRY,tc='white',bold=True)
Y=[7.7,6.0,4.3,2.6]
fb(XL,Y[0],5.8,H2,'2  Baseline: a Dummy estimator,\ncross-validated')
fb(XL,Y[1],5.8,H2,'3  Candidates: k-fold\ncross-validation, chosen scoring')
fb(XL,Y[2],5.8,H2,'4  Keep the best mean score\n(and look at the spread)')
fb(XL,Y[3],5.8,H2,'5  Refit the winner on the\nwhole training part',fc='#E3F4EA',ec=GREEN)
fb(5.6,0.7,10.4,0.85,'6  Score the test part once; report that number',fc='#E3F4EA',ec=GREEN)
A((5.2,11.85),(5.2,11.25)); A((XL,10.37),(XL,9.65)); A((XR,10.37),(XR,9.65),c=GREY)
A((XL,8.95),(XL,Y[0]+H2/2))
for k in range(3): A((XL,Y[k]-H2/2),(XL,Y[k+1]+H2/2))
A((XL,Y[3]-H2/2),(XL,1.13))
ax.plot([XR,XR],[8.95,1.5],color=GREY,lw=1.0,ls=(0,(4,2)),zorder=0); A((XR,1.9),(XR,1.13),c=GREY)
ax.text(XR+0.2,5.0,'untouched\nuntil\nstep 6',fontsize=8.8,color=GREY,ha='left',va='center',style='italic')
ax.plot([6.2,6.9,6.9,6.2],[Y[2],Y[2],Y[1],Y[1]],color=PUR,lw=1.0); arr(ax,(6.55,Y[1]),(6.2,Y[1]),c=PUR,lw=1.0,ms=8)
ax.text(7.05,(Y[1]+Y[2])/2,'change\nthe model,\nrepeat',fontsize=8.8,color=PUR,ha='left',va='center')
ax.set_xlim(0,11.4); ax.set_ylim(0.1,12.7); save2(fig,'f871.png',340)

NUM['fontcheck']=FC
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(FC))

# All figures of Chapter 7. Run: python3 fig/figs.py   (writes PNGs into ch07/ and numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Rectangle, FancyBboxPatch
from sklearn.datasets import (load_iris, load_diabetes, load_breast_cancer, load_digits, make_regression,
                              make_classification, make_blobs, make_moons, make_circles)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
NUM={}
GREENT='#E3F4EA'; PURT='#F1EAF7'; ORT='#FDF0E0'
def bx(ax,x,y,w,h,txt,fc=LB,ec=BLUE,fs=8.4,tc=NAVY,bold=False,r=0.08,ha='center'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0.0,rounding_size={r}',fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2 if ha=='center' else x+0.12,y+h/2,txt,ha=ha,va='center',fontsize=fs,color=tc,weight='bold' if bold else 'normal',linespacing=1.25)

# ---- 7.0-1 opener: the contract of an estimator
fig,ax=new(5.0,2.0); ax.set_xlim(0,10); ax.set_ylim(0,4.0)
bx(ax,0.0,1.2,1.9,1.5,'your data\n\n$\\mathbf{X}$   $(n\\times p)$\n$\\mathbf{y}$   $(n,)$',fc=LGREY,ec=GREY,fs=8.4)
arr(ax,(1.95,1.95),(3.05,1.95),c=GREEN,lw=1.4,ms=9); ax.text(2.5,2.2,'fit(X, y)',ha='center',fontsize=8.4,color=GREEN,weight='bold')
ax.add_patch(FancyBboxPatch((3.1,0.05),3.6,3.9,boxstyle='round,pad=0.0,rounding_size=0.1',fc='white',ec=NAVY,lw=1.3))
ax.text(4.9,3.65,'an estimator',ha='center',va='center',fontsize=9.2,color=NAVY,weight='bold')
bx(ax,3.25,2.0,3.3,1.3,'hyperparameters\nset by you when created\nC = 1.0,   max_iter = 100',fc=PURT,ec=PUR,fs=8.0)
bx(ax,3.25,0.2,3.3,1.65,'learned attributes\nset by fit, names end in _\ncoef_,  intercept_,\nclasses_,  n_features_in_',fc=GREENT,ec=GREEN,fs=8.0)
outs=[('predict(X_new)','labels or numbers $\\hat{y}$'),('predict_proba(X_new)','probabilities, $n\\times K$'),('transform(X_new)','new feature table'),('score(X, y)','one quality number')]
for k,(a_,b_) in enumerate(outs):
    yy=3.35-k*0.98; bx(ax,7.2,yy-0.4,2.8,0.8,f'{a_}\n{b_}',fc=LB,ec=BLUE,fs=7.6)
    arr(ax,(6.75,1.95),(7.15,yy),c=NAVY,lw=0.9,ms=7)
save(fig,'f700.png')

# ---- 7.2-1 the same matrix as dense table and as CSR
A=np.array([[0,0,3,0,0],[1,0,0,0,2],[0,0,0,0,0],[0,4,0,5,0]])
import scipy.sparse as sp
S=sp.csr_array(A); NUM['csr']={'data':S.data.tolist(),'indices':S.indices.tolist(),'indptr':S.indptr.tolist()}
pal=['#CFE0F7','#FBDDB8','#CDEBD8','#E4D6F0','#F7CFCB']
fig,ax=new(5.0,2.3); ax.set_xlim(0,10); ax.set_ylim(0.3,5.0)
cw,chh=0.62,0.62; x0,y0=0.55,4.0
ax.text(x0+2.5*cw,y0+0.62,'dense table $A$:  row $i$, column $j$',ha='center',fontsize=8.4,color=NAVY)
for j in range(5): ax.text(x0+j*cw+cw/2,y0+0.18,str(j),ha='center',fontsize=8,color=GREY)
k=0; cell={}
for i in range(4):
    ax.text(x0-0.2,y0-i*chh-chh/2,str(i),ha='right',va='center',fontsize=8,color=GREY)
    for j in range(5):
        v=A[i,j]; fc='white' if v==0 else pal[k%5]
        ax.add_patch(Rectangle((x0+j*cw,y0-(i+1)*chh),cw,chh,fc=fc,ec=GRID if v==0 else NAVY,lw=0.7))
        ax.text(x0+j*cw+cw/2,y0-(i+.5)*chh,str(v),ha='center',va='center',fontsize=8.4,color='#AAAAAA' if v==0 else NAVY)
        if v!=0: cell[(i,j)]=k; k+=1
ax.text(x0+2.5*cw,y0-4*chh-0.55,'5 of 20 entries are not zero',ha='center',fontsize=8.2,color=GREY)
# right: three arrays
xs=5.3; cw2=0.82
def strip(y,name,vals,cols=None,labels=None):
    ax.text(xs-0.12,y+0.3,name,ha='right',va='center',fontsize=8.4,color=NAVY,weight='bold')
    for q,v in enumerate(vals):
        ax.add_patch(Rectangle((xs+q*cw2,y),cw2,0.6,fc=cols[q] if cols else LGREY,ec=NAVY,lw=0.7)); ax.text(xs+q*cw2+cw2/2,y+0.3,str(v),ha='center',va='center',fontsize=8.4,color=NAVY)
        if labels: ax.text(xs+q*cw2+cw2/2,y+0.72,str(labels[q]),ha='center',fontsize=8,color=GREY)
ax.text(xs+2.5*cw2,4.72,'three arrays of the CSR format',ha='center',fontsize=8.4,color=NAVY)
strip(3.35,'data',S.data,pal,labels=range(5)); strip(2.3,'indices',S.indices,pal,labels=range(5))
strip(1.05,'indptr',S.indptr,None,labels=['row 0','row 1','row 2','row 3','end'])
ax.text(xs+2.5*cw2,0.5,'row $i$ owns positions indptr[$i$] up to indptr[$i$+1] − 1',ha='center',fontsize=8.0,color=GREY)
ax.text(xs-0.12,4.07,'position $k$',ha='right',va='center',fontsize=7.8,color=GREY)
save(fig,'f721.png')

# ---- 7.3-1 gallery of generators and loaders
fig,axs=plt.subplots(2,4,figsize=(5.0,2.5),dpi=240)
cl=[BLUE,OR,TEAL]; mk=['o','s','^']
def sc(ax,X,y,s=5):
    for c_ in np.unique(y): ax.scatter(X[y==c_,0],X[y==c_,1],s=s,color=cl[int(c_)%3],marker=mk[int(c_)%3],lw=0,alpha=0.85)
def clean(ax,t): ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t,fontsize=7.8,color=NAVY,pad=3)
X,y=make_regression(n_samples=60,n_features=1,noise=8,random_state=0); axs[0,0].scatter(X[:,0],y,s=5,color=BLUE,lw=0); clean(axs[0,0],'make_regression')
X,y=make_classification(n_samples=120,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=3); sc(axs[0,1],X,y); clean(axs[0,1],'make_classification')
X,y=make_blobs(n_samples=120,centers=3,random_state=0); sc(axs[0,2],X,y); clean(axs[0,2],'make_blobs')
X,y=make_moons(n_samples=120,noise=0.12,random_state=0); sc(axs[0,3],X,y); clean(axs[0,3],'make_moons')
X,y=make_circles(n_samples=120,noise=0.05,factor=0.5,random_state=0); sc(axs[1,0],X,y); clean(axs[1,0],'make_circles')
ir=load_iris(); sc(axs[1,1],ir.data[:,2:4],ir.target); clean(axs[1,1],'load_iris (petals)')
db=load_diabetes(); axs[1,2].scatter(db.data[:,2],db.target,s=3,color=BLUE,lw=0,alpha=0.7); clean(axs[1,2],'load_diabetes (bmi)')
dg=load_digits(); ax=axs[1,3]; ax.axis('off'); ax.set_title('load_digits',fontsize=7.8,color=NAVY,pad=3)
for q,idx in enumerate([0,1,2,3]):
    ins=ax.inset_axes([0.03+0.5*(q%2),0.52-0.5*(q//2),0.44,0.44]); ins.imshow(dg.images[idx],cmap='Greys'); ins.axis('off')
for a in axs.ravel(): a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
fig.tight_layout(w_pad=0.5,h_pad=0.8); save(fig,'f731.png')

# ---- 7.4-1 life of an estimator object
fig,ax=new(4.6,1.85); ax.set_xlim(0,10); ax.set_ylim(0,4.0)
bx(ax,0.0,1.9,2.8,1.9,'LogisticRegression\n(C=10.0)\n\nsettings only,\nno coef_',fc=PURT,ec=PUR,fs=7.8)
bx(ax,3.6,1.9,2.8,1.9,'the same object,\nfitted\n\nC = 10.0\ncoef_,  intercept_, ...',fc=GREENT,ec=GREEN,fs=7.8)
bx(ax,7.2,1.9,2.8,1.9,'twin = clone(model)\n\nC = 10.0\nno coef_',fc=PURT,ec=PUR,fs=7.8)
arr(ax,(2.85,2.85),(3.55,2.85),c=NAVY,lw=1.1,ms=8); ax.text(3.2,3.12,'fit',ha='center',fontsize=8.2,color=NAVY,weight='bold')
arr(ax,(6.45,2.85),(7.15,2.85),c=NAVY,lw=1.1,ms=8); ax.text(6.8,3.12,'clone',ha='center',fontsize=8.2,color=NAVY,weight='bold')
bx(ax,2.6,0.05,4.8,1.2,'set_params(C=0.1) changes C only;\nthe old coef_ stays until the next fit',fc=ORT,ec=OR,fs=7.6)
ax.plot([5.0,5.0],[1.28,1.87],color=OR,lw=1.0,ls='--')
save(fig,'f741.png')

# ---- 7.5-1 shapes returned by the methods
Xi,yi=load_iris(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xi,yi,test_size=0.25,random_state=0)
sc_=StandardScaler().fit(Xtr); Ztr,Zte=sc_.transform(Xtr),sc_.transform(Xte); clf=LogisticRegression().fit(Ztr,ytr)
four=Zte[:4]; P_=clf.predict_proba(four); Fd=clf.decision_function(four); Pr=clf.predict(four)
NUM['shapes']={'proba_row_sums':P_.sum(1).round(6).tolist(),'predict':Pr.tolist()}
fig,ax=new(5.0,2.1); ax.set_xlim(0,10); ax.set_ylim(0.5,4.8)
def grid(x0,y0,vals,cw_,title,shape,fc,ec,fmt='{:.2f}',fs=7.6):
    nr,nc=len(vals),len(vals[0])
    for i in range(nr):
        for j in range(nc):
            ax.add_patch(Rectangle((x0+j*cw_,y0-(i+1)*0.5),cw_,0.5,fc=fc,ec=ec,lw=0.7))
            v=vals[i][j]; ax.text(x0+j*cw_+cw_/2,y0-(i+.5)*0.5,fmt.format(v).replace('-','−') if not isinstance(v,(int,np.integer)) else str(v),ha='center',va='center',fontsize=fs,color=NAVY)
    ax.text(x0+nc*cw_/2,y0+0.62,title,ha='center',fontsize=7.8,color=NAVY,weight='bold',linespacing=1.15); ax.text(x0+nc*cw_/2,y0-nr*0.5-0.22,shape,ha='center',va='top',fontsize=7.8,color=GREY)
grid(0.05,3.7,[[int(v)] for v in Pr],0.55,'predict','shape (4,)',GREENT,GREEN)
grid(1.1,3.7,P_.round(2).tolist(),0.82,'predict_proba','shape (4, 3)',LB,BLUE)
grid(4.15,3.7,Fd.round(2).tolist(),0.9,'decision_function','shape (4, 3)',PURT,PUR)
grid(7.2,3.7,four.round(2).tolist(),0.68,'transform','shape (4, 4)',LGREY,GREY)
ax.text(1.1+1.23,0.95,'each row sums to 1',ha='center',fontsize=7.8,color=BLUE,style='italic'); ax.text(4.15+1.35,0.95,'raw scores, any sign',ha='center',fontsize=7.8,color=PUR,style='italic')
ax.text(0.33,0.95,'labels',ha='center',fontsize=7.8,color=GREEN,style='italic'); ax.text(7.2+1.36,0.95,'new features',ha='center',fontsize=7.8,color=GREY,style='italic')
ax.text(5.0,0.2,'four test flowers of iris;  columns of the middle two: setosa, versicolor, virginica',ha='center',fontsize=7.8,color=GREY)
save(fig,'f751.png')

# ---- 7.6-1 standardising four numbers
xv=np.array([2.,4,6,8]); mu=xv.mean(); sd=xv.std(); zv=(xv-mu)/sd; xt=10.; zt=(xt-mu)/sd
NUM['scaler']={'mu':float(mu),'sd':round(float(sd),4),'z':zv.round(4).tolist(),'z_test':round(float(zt),4)}
fig,axs=plt.subplots(2,1,figsize=(4.7,2.3),dpi=240,gridspec_kw={'hspace':1.0})
for ax,vals,vt,m_,s_,xl,lab,xlim in zip(axs,[xv,zv],[xt,zt],[mu,0],[sd,1],[(0,11.5),(-2.1,3.1)],['raw  $x$','standardised  $z$'],[(0,11.5),(-2.1,3.1)]):
    ax.set_xlim(*xl); ax.set_ylim(-1,1.3); ax.axhline(0,color=GRID,lw=1); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    ax.axvspan(m_-s_,m_+s_,color=LB,alpha=0.9,zorder=0); ax.axvline(m_,color=GREEN,lw=1.4,zorder=1)
    ax.scatter(vals,np.zeros(4),s=34,color=BLUE,edgecolor='white',lw=0.8,zorder=5); ax.scatter([vt],[0],s=34,facecolor='white',edgecolor=GREY,lw=1.3,zorder=5)
    for v in vals: ax.text(v,0.35,f'{v:g}' if ax is axs[0] else f'{v:.2f}'.replace('-','−'),ha='center',fontsize=8,color=BLUE)
    ax.text(vt,0.35,f'{vt:g}' if ax is axs[0] else f'{vt:.2f}',ha='center',fontsize=8,color=GREY)
    ax.set_xlabel(lab,fontsize=8.6,labelpad=1); ax.tick_params(labelsize=8)
axs[0].text(mu+0.1,-0.62,'mean 5',color=GREEN,fontsize=8.2,ha='left',va='center'); axs[0].text(mu+sd+0.08,-0.62,'+1 std = 7.24',color=NAVY,fontsize=8,ha='left',va='center')
axs[1].text(0.08,-0.62,'mean 0',color=GREEN,fontsize=8.2,ha='left',va='center'); axs[1].text(1.08,-0.62,'+1 std = 1',color=NAVY,fontsize=8,ha='left',va='center')
axs[0].set_title('filled: training numbers      open: a later number',fontsize=8,color=GREY,pad=4)
save(fig,'f761.png')

# ---- 7.7-1 pipeline fit and predict
fig,ax=new(5.0,2.9); ax.set_xlim(0,10); ax.set_ylim(0.3,6.4)
def col(x0,title,steps,endt):
    ax.text(x0+2.1,6.15,title,ha='center',fontsize=8.8,color=NAVY,weight='bold')
    cen=[5.35,3.95,2.5,1.05]; hh=[0.65,1.2,0.65,1.2]; fcs=[LGREY,LB,LGREY,GREENT]; ecs=[GREY,BLUE,GREY,GREEN]
    for q in range(4): bx(ax,x0,cen[q]-hh[q]/2,4.2,hh[q],steps[q],fc=fcs[q],ec=ecs[q],fs=7.9)
    for q in range(3): arr(ax,(x0+2.1,cen[q]-hh[q]/2-0.02),(x0+2.1,cen[q+1]+hh[q+1]/2+0.02),c=NAVY,lw=0.9,ms=7)
col(0.0,'pipe.fit(X_train, y_train)',['X_train, y_train','StandardScaler\nlearns mean_, scale_\nthen transforms X_train','Z_train  (scaled)','LogisticRegression\nlearns coef_, intercept_\nfrom Z_train, y_train'],'')
col(5.7,'pipe.predict(X_new)',['X_new','StandardScaler\nonly transforms,\nwith the stored mean_, scale_','Z_new  (scaled)','LogisticRegression\npredicts with the\nstored coef_, intercept_'],'')
ax.plot([4.55,5.45],[3.95,3.95],color=PUR,lw=1.0,ls='--'); ax.text(5.0,4.25,'stored',ha='center',fontsize=7.6,color=PUR,style='italic')
ax.plot([4.55,5.45],[1.05,1.05],color=PUR,lw=1.0,ls='--'); ax.text(5.0,1.35,'stored',ha='center',fontsize=7.6,color=PUR,style='italic')
ax.set_xlim(0,10.1); save(fig,'f771.png')

# ---- 7.8-1 the iris project
iris=load_iris(); Xa,ya=iris.data,iris.target
Xtr,Xte,ytr,yte=train_test_split(Xa,ya,test_size=0.25,random_state=0,stratify=ya)
mdl=make_pipeline(StandardScaler(),LogisticRegression()).fit(Xtr,ytr); pred=mdl.predict(Xte); cm=confusion_matrix(yte,pred)
wrong=np.flatnonzero(pred!=yte); NUM['iris']={'acc':round(float(mdl.score(Xte,yte)),4),'wrong':wrong.tolist(),'cm':cm.tolist()}
fig,axs=plt.subplots(1,2,figsize=(5.0,2.25),dpi=240,gridspec_kw={'width_ratios':[1.35,1]})
ax=axs[0]
for c_ in range(3):
    ax.scatter(Xtr[ytr==c_,2],Xtr[ytr==c_,3],s=9,color=cl[c_],marker=mk[c_],lw=0,alpha=0.45)
    ax.scatter(Xte[yte==c_,2],Xte[yte==c_,3],s=22,facecolor='none',edgecolor=cl[c_],marker=mk[c_],lw=1.1)
for i in wrong: ax.scatter(Xte[i,2],Xte[i,3],s=150,facecolor='none',edgecolor=RED,lw=1.5); ax.annotate('the one\nmistake',xy=(Xte[i,2],Xte[i,3]),xytext=(Xte[i,2]+0.55,Xte[i,3]-0.95),fontsize=8,color=RED,arrowprops=dict(arrowstyle='-',color=RED,lw=0.7))
ax.set_xlabel('petal length (cm)',fontsize=8.6); ax.set_ylabel('petal width (cm)',fontsize=8.6); ax.tick_params(labelsize=8); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
for c_,(nm,col_) in enumerate(zip(['setosa','versicolor','virginica'],cl)): ax.text(0.03,0.97-0.085*c_,nm,transform=ax.transAxes,fontsize=8,color=col_,va='top')
ax.set_title('training (faint) and test (open)',fontsize=8.6,color=NAVY)
ax=axs[1]; ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max()*1.4)
for i in range(3):
    for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color=NAVY if cm[i,j]<9 else 'white')
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(['set','ver','vir'],fontsize=8); ax.set_yticklabels(['set','ver','vir'],fontsize=8)
ax.set_xlabel('predicted',fontsize=8.6); ax.set_ylabel('true',fontsize=8.6); ax.set_title('confusion matrix',fontsize=8.6,color=NAVY)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(w_pad=1.2); save(fig,'f781.png')

# ---- 7.9-1 spread of test accuracy over seeds
acc=[]; 
for seed in range(200):
    a,b_,c_,d_=train_test_split(Xa,ya,test_size=0.25,random_state=seed)
    acc.append(make_pipeline(StandardScaler(),LogisticRegression()).fit(a,c_).score(b_,d_))
acc=np.array(acc); NUM['seeds']={'mean':round(float(acc.mean()),4),'sd':round(float(acc.std()),4),'min':round(float(acc.min()),4),'max':round(float(acc.max()),4),'seed0':round(float(acc[0]),4)}
vals,cnt=np.unique(np.round(acc,4),return_counts=True)
fig,ax=plt.subplots(figsize=(4.0,2.15),dpi=240)
ax.axvspan(acc.mean()-acc.std(),acc.mean()+acc.std(),color=LB,zorder=0)
ax.bar(vals,cnt,width=0.0055,color=BLUE,zorder=3); ax.axvline(acc.mean(),color=GREEN,lw=1.4,zorder=4)
ax.plot([acc[0]],[cnt[list(vals).index(round(acc[0],4))]+3],'v',color=RED,ms=6,zorder=6); ax.text(acc[0]+0.007,cnt[list(vals).index(round(acc[0],4))]+1.5,'random_state = 0',fontsize=8,color=RED,va='center',zorder=6)
ax.text(acc.mean()-0.004,max(cnt)*1.05,'mean',fontsize=8,color=GREEN,ha='right',va='top'); ax.text(acc.mean()+acc.std()+0.003,max(cnt)*0.55,'±1 s.d.',fontsize=8,color=NAVY,ha='left')
ax.set_xlabel('test accuracy of one split',fontsize=8.6); ax.set_ylabel('number of splits',fontsize=8.6); ax.tick_params(labelsize=8); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_xlim(0.85,1.03); ax.set_ylim(0,76)
save(fig,'f791.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

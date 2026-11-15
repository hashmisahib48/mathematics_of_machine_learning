# Digits figures of Chapter 44. Run: python3 fig/figs_digits.py
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE, trustworthiness
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
X,y=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
E=np.load('verify/digits_embeddings.npz'); cols=plt.cm.tab10(np.arange(10))
# ---- 44.14-2 six embeddings of the digits
fig,axs=plt.subplots(2,3,figsize=(4.6,3.35),dpi=240); knn=KNeighborsClassifier(5); NUM['digits']={}
for ax,name in zip(axs.ravel(),['PCA','MDS (SMACOF)','Isomap','LLE','Laplacian','t-SNE']):
    Z=E[name]; T=trustworthiness(Xtr,Z,n_neighbors=10); cv=cross_val_score(knn,Z,ytr,cv=5).mean(); NUM['digits'][name]={'T10':float(T),'cv':float(cv)}
    lo,hi=np.percentile(Z,[1,99],axis=0); pad=(hi-lo)*0.15
    ax.scatter(Z[:,0],Z[:,1],c=ytr,cmap='tab10',vmin=-0.5,vmax=9.5,s=3,lw=0); ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]); ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]); clean(ax)
    ax.set_title(name,fontsize=9,color=NAVY,pad=2); ax.text(0.5,-0.02,f'T(10) = {T:.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color=GREY)
import matplotlib.lines as ml
h=[ml.Line2D([],[],marker='o',ls='',color=cols[d],ms=4,label=str(d)) for d in range(10)]
fig.legend(handles=h,loc='lower center',ncol=10,fontsize=8.5,handletextpad=0.1,columnspacing=0.8,frameon=False,bbox_to_anchor=(0.5,-0.015))
fig.subplots_adjust(left=0.01,right=0.99,top=0.95,bottom=0.15,wspace=0.05,hspace=0.32); sv(fig,'f4411.png',440)
# ---- 44.13-1 t-SNE: perplexity gallery and reading the plot
sub=slice(0,800); fig,axs=plt.subplots(1,4,figsize=(4.6,1.45),dpi=240); NUM['perp']={}
for ax,p in zip(axs,(2,5,30,100)):
    ts=TSNE(perplexity=p,init='pca',random_state=0).fit(Xtr[sub]); Z=ts.embedding_; T=trustworthiness(Xtr[sub],Z,n_neighbors=10)
    NUM['perp'][p]={'kl':float(ts.kl_divergence_),'T10':float(T),'n_iter':int(ts.n_iter_)}
    ax.scatter(Z[:,0],Z[:,1],c=ytr[sub],cmap='tab10',vmin=-0.5,vmax=9.5,s=2.2,lw=0); clean(ax); ax.set_title(f'perplexity {p}',fontsize=8.8,color=PUR,pad=2)
    ax.text(0.5,-0.02,f'KL = {ts.kl_divergence_:.2f}',transform=ax.transAxes,ha='center',va='top',fontsize=8.5,color=GREY)
fig.subplots_adjust(left=0.01,right=0.99,top=0.88,bottom=0.12,wspace=0.06); sv(fig,'f4412.png',440)
json.dump(NUM,open('verify/fig_numbers_digits.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize_digits.json','w'),indent=1); print('ok')

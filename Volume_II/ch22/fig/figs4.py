# Figure of the case study (22.13). Run: python3 fig/figs4.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
import matplotlib.text as mtext
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}
X,y=load_breast_cancer(return_X_y=True); X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
Cs=2.0**np.arange(-5,10,2); gs_=2.0**np.arange(-15,2,2)
gs=GridSearchCV(make_pipeline(StandardScaler(),SVC()),{'svc__C':Cs,'svc__gamma':gs_},cv=5).fit(X_tr,y_tr)
H=gs.cv_results_['mean_test_score'].reshape(len(Cs),len(gs_)).T   # rows gamma, cols C
bi=np.unravel_index(H.argmax(),H.shape)   # first maximum in row-major order
# all cells that tie the best
ties=np.argwhere(H>=H.max()-1e-12)
NUM['grid']={'best_cv':round(float(H.max()),4),'n_ties':int(len(ties)),'min_cv':round(float(H.min()),4),'best_C':float(gs.best_params_['svc__C']),'best_gamma':float(gs.best_params_['svc__gamma']),
  'default_cell_cv':round(float(cross_val_score(make_pipeline(StandardScaler(),SVC()),X_tr,y_tr,cv=5).mean()),4)}
Cc=2.0**np.arange(-5,10,1.0)
raw=[cross_val_score(SVC(C=c),X_tr,y_tr,cv=5).mean() for c in Cc]; scl=[cross_val_score(make_pipeline(StandardScaler(),SVC(C=c)),X_tr,y_tr,cv=5).mean() for c in Cc]
NUM['curves']={'C':[float(v) for v in Cc],'raw':[round(float(v),4) for v in raw],'scaled':[round(float(v),4) for v in scl],'best_raw':round(float(max(raw)),4),'best_raw_C':float(Cc[int(np.argmax(raw))]),'best_scaled':round(float(max(scl)),4),'best_scaled_C':float(Cc[int(np.argmax(scl))])}
fig,axs=plt.subplots(1,2,figsize=(4.9,2.6),dpi=240,gridspec_kw={'width_ratios':[1.1,1]}); ax=axs[0]
ax.imshow(H,origin='lower',cmap='Blues',vmin=0.6,vmax=1.0,aspect='auto')
ax.scatter(*[ties[:,1],ties[:,0]],marker='*',s=60,color=OR,edgecolor='white',lw=0.5,zorder=5)
dg=list(gs_).index(2.0**-5) if 2.0**-5 in gs_ else None
ax.set_xticks(range(0,len(Cs),2)); ax.set_xticklabels([f'$2^{{{int(np.log2(v))}}}$' for v in Cs[::2]],fontsize=8); ax.set_yticks(range(0,len(gs_),2)); ax.set_yticklabels([f'$2^{{{int(np.log2(v))}}}$' for v in gs_[::2]],fontsize=8)
ax.set_xlabel('$C$'); ax.set_ylabel('$\\gamma$'); ax.set_title('5-fold accuracy, scaled',fontsize=9,color=NAVY)
ax=axs[1]; ax.semilogx(Cc,scl,'-o',color=GREEN,ms=3,lw=1.5,label='scaled features'); ax.semilogx(Cc,raw,'--s',color=GREY,ms=3,lw=1.3,label='raw features')
ax.set_xlabel('$C$  (default $\\gamma$)'); ax.set_ylabel('5-fold accuracy'); ax.legend(loc='lower right',fontsize=8.5); ax.grid(color=GRID,lw=0.5); ax.set_title('scaling matters',fontsize=9,color=NAVY)
fig.tight_layout(w_pad=1.0)
fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
fig.savefig('f2213.png',dpi=240,bbox_inches='tight',pad_inches=0.04,facecolor='white')
from PIL import Image; w=Image.open('f2213.png').size[0]/240; NUM['textsize']={'fig_in':round(w,2),'min_font_printed':round(fs*(470/96)/w,1)}
json.dump(NUM,open('verify/fig_numbers4.json','w'),indent=1); print(json.dumps(NUM))

# Problem figures of Chapter 53. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectFdr, SelectFpr, SelectFwe, SelectFromModel, SequentialFeatureSelector, chi2, f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from bcdata import load
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.15): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); fig.tight_layout(pad=0.4); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*105,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
r4=lambda v:[round(float(x),4) for x in np.ravel(v)]
Xa,Xb,ya,yb,names=load()

# 53A.1 three-class F
x=np.arange(1.,10); y=np.repeat([0,1,2],3); Fv,pv=f_classif(x[:,None],y)
A['53A.1']={'F':round(float(Fv[0]),4),'p':round(float(pv[0]),6),'SSB':54,'SSW':6}
fig,ax=small(2.0,0.81); cols=[BLUE,OR,GREEN]
for k in range(3): ax.plot(x[y==k],np.zeros(3),'o',color=cols[k],ms=5)
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlabel('value of the feature'); ax.set_xticks([1,3,5,7,9]); sv(fig,'p_53A1.png')

# 53A.2 chi2 three classes
c=np.array([2,2,2,2,1,1,0,0,0,1,0,1.]); z=np.repeat([0,1,2],4); ch,pc=chi2(c[:,None],z)
O=[c[z==k].sum() for k in range(3)]
A['53A.2']={'observed':O,'expected':[c.sum()/3]*3,'chi2':round(float(ch[0]),4),'p':round(float(pc[0]),4),'df':2}
fig,ax=small(2.0,0.85); ax.bar(np.arange(3)-0.18,O,0.36,color=BLUE,label='observed'); ax.bar(np.arange(3)+0.18,[c.sum()/3]*3,0.36,color=LGREY,edgecolor=GREY,label='expected'); ax.set_xticks(range(3)); ax.set_xticklabels(['class 0','class 1','class 2']); ax.set_ylabel('feature total'); ax.legend(fontsize=9,handlelength=1,loc='upper center',ncol=2,columnspacing=0.8,borderaxespad=0.1); ax.set_ylim(0,14.5); grid(ax); sv(fig,'p_53A2.png')

# 53A.3 BH, Bonferroni, FPR
p=np.array([0.51,0.002,0.043,0.20,0.021,0.08,0.009,0.034]); m=8; al=0.10
ps=np.sort(p); line=al*np.arange(1,m+1)/m; ks=int(np.flatnonzero(ps<=line).max()+1)
f=lambda S:S(lambda X,yy:(np.zeros(m),p),alpha=al).fit(np.zeros((9,m)),np.zeros(9)).get_support().sum()
A['53A.3']={'sorted_p':ps.tolist(),'line':r4(line),'k_star':ks,'BH_kept':int(f(SelectFdr)),'Bonferroni_kept':int(f(SelectFwe)),'FPR_kept':int(f(SelectFpr))}
fig,ax=small(2.0,1.05); k=np.arange(1,m+1); ax.plot(k,ps,'o',color=BLUE,ms=4); ax.plot(k,line,'--',color=GREEN,lw=1.3); ax.axhline(al,color=RED,lw=1); ax.axhline(al/m,color=PUR,lw=1,ls=':'); ax.set_xlabel('rank $k$'); ax.set_ylabel('$p_{(k)}$'); ax.set_ylim(0,0.2); ax.set_xticks([1,4,8]); ax.set_yticks([0,0.1,0.2]); grid(ax); sv(fig,'p_53A3.png')

# 53B.1 F misses a U-shaped relation, mutual information does not
rng=np.random.RandomState(1); xu=rng.uniform(-1,1,300); xn=rng.uniform(-1,1,300); yu=(np.abs(xu)>0.5).astype(int)
Xu=np.c_[xu,xn]; Fu,_=f_classif(Xu,yu); mu=mutual_info_classif(Xu,yu,random_state=0)
A['53B.1']={'F':r4(Fu),'MI':r4(mu)}
fig,ax=small(2.0,0.94); ax.plot(xu[yu==0],xn[yu==0],'o',color=BLUE,ms=2,alpha=0.7); ax.plot(xu[yu==1],xn[yu==1],'s',color=OR,ms=2,alpha=0.7); ax.set_xlabel('$x$ (class: $|x|>0.5$)'); ax.set_ylabel('noise'); ax.set_yticks([-1,1]); sv(fig,'p_53B1.png')

# 53B.2 RFE needs scaled columns
lrm=lambda: LogisticRegression(max_iter=10000)
raw=RFE(lrm(),n_features_to_select=5).fit(Xa,ya).get_support(); Zs=StandardScaler().fit(Xa); sc=RFE(lrm(),n_features_to_select=5).fit(Zs.transform(Xa),ya).get_support()
A['53B.2']={'raw':np.flatnonzero(raw).tolist(),'scaled':np.flatnonzero(sc).tolist(),'overlap':int((raw&sc).sum()),'raw_columns_std':[round(float(Xa[:,j].std()),3) for j in np.flatnonzero(raw)]}
fig,ax=small(2.0,0.81)
for r_,(msk,col,mk) in enumerate(((raw,RED,'o'),(sc,GREEN,'s'))): idx=np.flatnonzero(msk); ax.plot(idx+1,np.full(len(idx),r_),mk,color=col,ms=5)
ax.set_yticks([0,1]); ax.set_yticklabels(['raw','scaled']); ax.set_ylim(-0.6,1.6); ax.set_xlim(0,31); ax.set_xlabel('column kept (1 to 30)'); sv(fig,'p_53B2.png')

# 53C.1 exclusive-or
rng=np.random.RandomState(2); u=rng.uniform(-1,1,(200,2)); yx=((u[:,0]>0)^(u[:,1]>0)).astype(int); Fx,_=f_classif(u,yx)
A['53C.1']={'F_each':r4(Fx)}
fig,ax=small(2.0,1.3); ax.plot(u[yx==0,0],u[yx==0,1],'o',color=BLUE,ms=2.4); ax.plot(u[yx==1,0],u[yx==1,1],'s',color=OR,ms=2.4); ax.axhline(0,color=GREY,lw=0.6); ax.axvline(0,color=GREY,lw=0.6); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_xticks([]); ax.set_yticks([]); sv(fig,'p_53C1.png')

# 53C.2 stability of an L1 selection under resampling
Zt=StandardScaler().fit_transform(Xa); freq=np.zeros(30); B=40
for b in range(B):
    idx=np.random.RandomState(b).randint(0,len(ya),len(ya)); freq+=SelectFromModel(LogisticRegression(l1_ratio=1,C=0.05,solver='liblinear',random_state=0)).fit(Zt[idx],ya[idx]).get_support()
freq/=B; ordr=np.argsort(-freq)
A['53C.2']={'freq_top8':r4(freq[ordr[:8]]),'n_always':int((freq==1).sum()),'n_ever':int((freq>0).sum())}
fig,ax=small(2.0,0.94); ax.bar(np.arange(30),freq[ordr],color=PUR,width=0.8); ax.set_xticks([]); ax.set_ylabel('share of fits'); ax.set_ylim(0,1.05); grid(ax); sv(fig,'p_53C2.png')

# 53C.3 leakage summary from the figure of Section 53.5
bb=json.load(open('verify/fig_numbers.json'))['bias']; lk,hn=bb['1000']
A['53C.3']={'leaky_m1000':round(lk,3),'honest_m1000':round(hn,3)}
fig,ax=small(2.0,0.85); ax.bar([0,1],[lk,hn],color=[RED,GREEN],width=0.55); ax.axhline(0.5,color=GREY,ls=':',lw=1); ax.set_xticks([0,1]); ax.set_xticklabels(['before CV','inside CV']); ax.set_ylabel('accuracy'); ax.set_ylim(0,1); sv(fig,'p_53C3.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

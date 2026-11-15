# Problem figures and answers of Chapter 7. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('ignore')
from figstyle import *
from matplotlib.patches import Rectangle, FancyBboxPatch
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
A={}
PURT='#F1EAF7'; GREENT='#E3F4EA'; ORT='#FDF0E0'
def small(w=2.4,h=1.7): return plt.subplots(figsize=(w,h),dpi=240)
def bx(ax,x,y,w,h,txt,fc=LB,ec=BLUE,fs=9,tc=NAVY):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.0,rounding_size=0.07',fc=fc,ec=ec,lw=0.9)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,color=tc,linespacing=1.2)
# ---- 7A.1 CSR
M=np.array([[5,0,0,6,0],[0,0,8,0,0],[0,1,0,0,3]]); S=sp.csr_array(M)
A['7A.1']={'data':S.data.tolist(),'indices':S.indices.tolist(),'indptr':S.indptr.tolist(),'stored_fraction':'%d/%d'%(S.nnz,M.size)}
pal=['#CFE0F7','#FBDDB8','#CDEBD8','#E4D6F0','#F7CFCB']
fig,ax=new(2.4,1.7); ax.set_xlim(0,5.4); ax.set_ylim(0,3.8); k=0; cw=0.9
for j in range(5): ax.text(0.5+j*cw+cw/2,3.45,str(j),ha='center',fontsize=9,color=GREY)
for i in range(3):
    ax.text(0.4,2.8-i*0.9-0.1,str(i),ha='right',va='center',fontsize=9,color=GREY)
    for j in range(5):
        v=M[i,j]; fc='white' if v==0 else pal[k%5]
        ax.add_patch(Rectangle((0.5+j*cw,2.8-i*0.9-0.55),cw,0.9,fc=fc,ec=GRID if v==0 else NAVY,lw=0.7)); ax.text(0.5+j*cw+cw/2,2.8-i*0.9-0.1,str(v),ha='center',va='center',fontsize=9.5,color='#AAAAAA' if v==0 else NAVY)
        if v: k+=1
ax.text(3.0,0.18,'the matrix $B$',ha='center',fontsize=9,color=GREY); save(fig,'p_7A1.png')
# ---- 7A.2 scaler
v=np.array([0.,2,2,8]); mu=v.mean(); sd=v.std(); z=(v-mu)/sd
A['7A.2']={'mean':float(mu),'std':float(sd),'z':z.round(4).tolist(),'z_12':float((12-mu)/sd),'std_n_minus_1':round(float(v.std(ddof=1)),4)}
fig,ax=small(); ax.axhline(0,color=GRID,lw=1); ax.scatter(v,[0,0,0.22,0],s=40,color=BLUE,edgecolor='white',zorder=5); ax.axvline(mu,color=GREEN,lw=1.5)
ax.axvspan(mu-sd,mu+sd,color=LB,zorder=0); ax.text(mu+0.15,-0.55,'mean',color=GREEN,fontsize=9); ax.set_ylim(-1,1); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_xlim(-1,9.5); ax.tick_params(labelsize=9); ax.set_xlabel('$x$',fontsize=9.5,labelpad=1)
ax.text(7.2,0.45,'four values',fontsize=9,color=BLUE); fig.subplots_adjust(bottom=0.28); save(fig,'p_7A2.png')
# ---- 7A.3 softmax
f=np.array([2.,0.,-1.]); p=softmax(f); A['7A.3']={'exp':np.exp(f).round(3).tolist(),'sum_exp':round(float(np.exp(f).sum()),3),'p':p.round(3).tolist(),'p_shift_+5':softmax(f+5).round(3).tolist(),'argmax':int(f.argmax())}
fig,ax=small(); ax.bar([0,1,2],f,width=0.55,color=[PUR if a>=0 else RED for a in f]); ax.axhline(0,color=GREY,lw=0.8); ax.set_xticks([0,1,2]); ax.set_xticklabels(['$f_1$','$f_2$','$f_3$'],fontsize=9.5); ax.tick_params(labelsize=9)
for a,b in zip([0,1,2],f): ax.text(a,b+(0.15 if b>=0 else -0.15),f'{b:g}'.replace('-','−'),ha='center',va='bottom' if b>=0 else 'top',fontsize=9)
ax.set_ylim(-1.8,2.8); ax.set_ylabel('score',fontsize=9.5); fig.subplots_adjust(left=0.2,bottom=0.17); save(fig,'p_7A3.png')
# ---- 7A.4 split sizes
res={}
for n,ts in [(150,0.2),(150,0.25),(442,0.25)]:
    a,b_=train_test_split(np.arange(n),test_size=ts,random_state=0); res[f'{n},{ts}']=[len(a),len(b_)]
A['7A.4']=res
fig,ax=small(2.4,1.3); ax.set_xlim(0,10); ax.set_ylim(0,3); ax.axis('off')
ax.add_patch(Rectangle((0,1.1),7.47,0.9,fc=LB,ec=BLUE)); ax.add_patch(Rectangle((7.47,1.1),2.53,0.9,fc='#E9EDF3',ec=GREY))
ax.text(3.7,1.55,'train',ha='center',va='center',fontsize=9.5,color=NAVY); ax.text(8.73,1.55,'test',ha='center',va='center',fontsize=9.5,color=NAVY)
ax.text(5,2.5,'$n$ = 150 rows, test_size = 0.25',ha='center',fontsize=9,color=NAVY); ax.text(5,0.4,'how many rows in each part?',ha='center',fontsize=9,color=GREY); save(fig,'p_7A4.png')
# ---- 7B.1 wine scales
w=load_wine(as_frame=True); X=w.data; sdv=X.std(ddof=0)
A['7B.1']={'shape':list(X.shape),'class_counts':np.bincount(w.target).tolist(),'max_feature':sdv.idxmax(),'max_std':round(float(sdv.max()),3),'min_feature':sdv.idxmin(),'min_std':round(float(sdv.min()),4),'ratio':round(float(sdv.max()/sdv.min()),0)}
fig,ax=small(); o=np.sort(sdv.values); ax.plot(range(13),o,'o-',color=BLUE,ms=4,lw=1); ax.set_yscale('log'); ax.set_xticks([]); ax.tick_params(labelsize=8.5); ax.set_ylabel('std of feature',fontsize=9.5); ax.set_xlabel('13 wine features, sorted',fontsize=9.5,labelpad=2)
ax.text(11.6,o[-1]*0.55,f'{o[-1]:.0f}',ha='right',fontsize=9,color=NAVY); ax.text(0.6,o[0]*0.62,f'{o[0]:.2f}',ha='left',va='center',fontsize=9,color=NAVY); ax.set_ylim(0.03,1000); ax.set_xlim(-0.7,12.7); fig.subplots_adjust(left=0.22,bottom=0.2); save(fig,'p_7B1.png')
# ---- 7B.2 seeds on wine
Xw,yw=load_wine(return_X_y=True); acc=[]
for s in range(200):
    a,b_,c_,d_=train_test_split(Xw,yw,test_size=0.25,random_state=s); acc.append(make_pipeline(StandardScaler(),LogisticRegression()).fit(a,c_).score(b_,d_))
acc=np.array(acc); m_=len(d_); A['7B.2']={'mean':round(float(acc.mean()),4),'sd':round(float(acc.std()),4),'min':round(float(acc.min()),4),'max':round(float(acc.max()),4),'m_test':m_,'binomial_sd':round(float(np.sqrt(acc.mean()*(1-acc.mean())/m_)),4)}
vals,cnt=np.unique(np.round(acc,4),return_counts=True)
fig,ax=small(); ax.bar(vals,cnt,width=0.012,color=BLUE); ax.axvline(acc.mean(),color=GREEN,lw=1.5); ax.set_xlabel('test accuracy (wine)',fontsize=9.5,labelpad=2); ax.set_ylabel('splits',fontsize=9.5); ax.tick_params(labelsize=8.5); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True); fig.subplots_adjust(left=0.2,bottom=0.22); save(fig,'p_7B2.png')
# ---- 7C.1 leakage
fig,ax=small(); ax.set_xlim(-0.3,10.7); ax.set_ylim(-1.3,1.6); ax.axhline(0,color=GRID,lw=1)
tr=np.array([2.,4,6,8]); ax.scatter(tr,np.zeros(4),s=40,color=BLUE,edgecolor='white',zorder=5); ax.scatter([10],[0],s=40,facecolor='white',edgecolor=GREY,lw=1.4,zorder=5)
ax.plot([5,5],[-0.5,0.5],color=GREEN,lw=2); ax.plot([6,6],[-0.5,0.5],color=RED,lw=2,ls=(0,(2,1)))
ax.text(4.9,0.62,'mean of\ntraining',ha='right',fontsize=8.6,color=GREEN,va='bottom'); ax.text(6.1,0.62,'mean with\nthe test point',ha='left',fontsize=8.6,color=RED,va='bottom')
ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.tick_params(labelsize=9); ax.set_xlabel('$x$',fontsize=9.5,labelpad=1); fig.subplots_adjust(bottom=0.25); save(fig,'p_7C1.png')
# ---- 7C.2 init vs fit
fig,ax=new(2.4,1.7); ax.set_xlim(0,10); ax.set_ylim(0,7)
bx(ax,0.0,3.9,10,2.9,'__init__\nonly stores the arguments',fc=PURT,ec=PUR,fs=9.2); bx(ax,0.0,0.2,10,2.9,'fit\ncreates names ending in _',fc=GREENT,ec=GREEN,fs=9.2)
save(fig,'p_7C2.png')
# ---- 7B.3 softmax with large scores
f3=np.array([1000.,1001,1002])
with np.errstate(over='ignore',invalid='ignore'):
    naive=np.exp(f3)/np.exp(f3).sum()
shift=np.exp(f3-f3.max())/np.exp(f3-f3.max()).sum()
A['7B.3']={'naive':[str(v) for v in naive],'shifted':[round(float(v),4) for v in shift],'scipy':[round(float(v),4) for v in softmax(f3)],'equal':bool(np.allclose(shift,softmax(f3)))}
fig,ax=small(2.6,1.3); ax.bar([1,2,3],shift,color=[BLUE,OR,GREEN],width=0.6)
for k_,v_ in enumerate(shift): ax.text(k_+1,v_+0.02,f'{v_:.3f}',ha='center',fontsize=9)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['1000','1001','1002']); ax.set_xlabel('score $f_k$',fontsize=9.5,labelpad=2); ax.set_ylabel('$P_k$',fontsize=9.5); ax.set_ylim(0,0.85); ax.set_yticks([0,0.4,0.8]); ax.tick_params(labelsize=9); fig.subplots_adjust(left=0.2,bottom=0.3); save(fig,'p_7B3.png')
# ---- 7C.3 one flower apart: the spread of iris accuracy over 200 splits
from sklearn.datasets import load_iris
Xi,yi=load_iris(return_X_y=True); acc_i=[]
for s in range(200):
    a,b_,c_,d_=train_test_split(Xi,yi,test_size=0.25,random_state=s); acc_i.append(make_pipeline(StandardScaler(),LogisticRegression()).fit(a,c_).score(b_,d_))
acc_i=np.array(acc_i); vals,cnt=np.unique(np.round(acc_i,4),return_counts=True)
A['7C.3']={'mean':round(float(acc_i.mean()),4),'sd':round(float(acc_i.std()),4),'counts':dict(zip([str(v) for v in vals],[int(c) for c in cnt]))}
fig,ax=small(); ax.bar(vals,cnt,width=0.018,color=LB if 'LB' in globals() else BLUE,ec=BLUE)
for v_,lab_,c_ in ((0.9737,'A',GREEN),(0.9474,'B',OR)): ax.annotate(lab_,(v_,cnt[list(np.round(vals,4)).index(v_)]),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9.5,color=c_,weight='bold')
ax.set_xlabel('test accuracy (iris), 200 splits',fontsize=9.5,labelpad=2); ax.set_ylabel('splits',fontsize=9.5); ax.set_ylim(0,80); ax.tick_params(labelsize=8.5); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True); fig.subplots_adjust(left=0.2,bottom=0.24); save(fig,'p_7C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1,default=str); print(json.dumps(A,default=str))

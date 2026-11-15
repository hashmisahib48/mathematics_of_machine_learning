# Problem figures of Chapter 24. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tree_scratch import MyTree
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; CHK={}
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def gini(c): p=np.asarray(c,float)/np.sum(c); return float(1-np.sum(p**2))
def ent(c): p=np.asarray(c,float)/np.sum(c); p=p[p>0]; return float(-np.sum(p*np.log2(p)))
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# 24A.1 impurity of a three-class node
c=[6,4,2]; A['24A.1']={'gini':round(gini(c),4),'entropy':round(ent(c),4),'misclass':round(1-6/12,4)}
fig,ax=small(); ax.bar([1,2,3],np.array(c)/12,color=[BLUE,OR,GREEN],width=0.6)
for k,v in enumerate(c): ax.text(k+1,v/12+0.02,f'{v}/12',ha='center',fontsize=9,color=NAVY)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['A','B','C']); ax.set_ylim(0,0.65); ax.set_ylabel('share $p_k$'); ax.set_xlabel('class'); grid(ax); sv(fig,'p_24A1.png')

# 24A.2 best split in one dimension
x2=np.arange(1,9); y2=np.array([0,0,1,0,0,1,1,1]); rows=[]
for t in x2[:-1]+0.5:
    L,R=y2[x2<=t],y2[x2>t]; rows.append((float(t),round((len(L)*gini(np.bincount(L,minlength=2))+len(R)*gini(np.bincount(R,minlength=2)))/len(y2),4)))
best=min(rows,key=lambda r:r[1]); A['24A.2']={'table':rows,'best':best,'unique':bool(sum(abs(r[1]-best[1])<1e-12 for r in rows)==1)}
fig,ax=small(2.4,1.2)
for xv,yv in zip(x2,y2): ax.scatter(xv,0,s=45,marker='o' if yv==0 else 's',color=BLUE if yv==0 else OR,edgecolor='white',lw=0.7,zorder=5)
ax.axhline(0,color=GREY,lw=0.8); ax.set_yticks([]); ax.set_xticks(x2); ax.set_xlabel('$x$'); ax.set_ylim(-0.5,0.5); ax.spines['left'].set_visible(False); sv(fig,'p_24A2.png')

# 24A.3 a regression split
x3=np.arange(1,6.); y3=np.array([2,3,10,11,12.]); sse=lambda v:float(np.sum((v-v.mean())**2)); tab=[]
for t in x3[:-1]+0.5: tab.append((float(t),round(sse(y3[x3<=t])+sse(y3[x3>t]),3)))
b3=min(tab,key=lambda r:r[1]); A['24A.3']={'table':tab,'best':b3,'SST':round(sse(y3),3),'means':[float(y3[x3<=b3[0]].mean()),float(y3[x3>b3[0]].mean())],'R2':round(1-b3[1]/sse(y3),4)}
fig,ax=small(); ax.scatter(x3,y3,s=34,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xticks(x3); ax.set_ylim(0,14); grid(ax); sv(fig,'p_24A3.png')

# 24A.4 weakest link: R(t) of the nodes
Rt={0:0.20,1:0.08,2:0.04,3:0.02,4:0.01}; kids={0:(1,2),1:(3,4)}
def leaves(t): return [t] if t not in kids else sum((leaves(c) for c in kids[t]),[])
def alpha(t,kd): 
    lv=[]; 
    def lf(u): 
        return [u] if u not in kd else sum((lf(c) for c in kd[u]),[])
    lv=lf(t); return (Rt[t]-sum(Rt[l] for l in lv))/(len(lv)-1)
al1={t:round(alpha(t,kids),4) for t in kids}; kids2={0:(1,2)}; al2={0:round(alpha(0,kids2),4)}
A['24A.4']={'alpha_first':al1,'alpha_second':al2,'best_at_0.06':'2 leaves (t1 pruned)'}
fig,ax=new(2.4,1.75)
P={0:(1.5,2.0),1:(0.8,1.0),2:(2.2,1.0),3:(0.3,0.0),4:(1.3,0.0)}
for a_,b_ in [(0,1),(0,2),(1,3),(1,4)]: ax.plot([P[a_][0],P[b_][0]],[P[a_][1],P[b_][1]],color=GREY,lw=0.9,zorder=1)
for t,(px,py) in P.items(): box(ax,px-0.4,py-0.27,0.8,0.54,f't{t}\n{Rt[t]:.2f}',fc=LB if t in kids else '#E3F4EA',ec=BLUE if t in kids else GREEN,fs=8.5)
ax.set_xlim(-0.3,2.8); ax.set_ylim(-0.4,2.4); sv(fig,'p_24A4.png')

# 24B.1 entropy in the scratch tree
class MyEntropyTree(MyTree):
    def _gini(self,n_k):
        p=n_k/n_k.sum(axis=-1,keepdims=True); q=np.where(p>0,p,1.0); return -(p*np.log2(q)).sum(axis=-1)
X,y=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y)
res={}
for d in (1,2,3):
    m=MyEntropyTree(max_depth=d).fit(Xtr,ytr); r=DTC(criterion='entropy',max_depth=d,random_state=0).fit(Xtr,ytr)
    res[d]={'same':bool(np.array_equal(m.predict(Xte),r.predict(Xte))),'acc':round(float(r.score(Xte,yte)),4)}
A['24B.1']=res
fig,ax=small(); p=np.linspace(0.001,0.999,300); ax.plot(p,-(p*np.log2(p)+(1-p)*np.log2(1-p)),color=PUR,lw=1.6,label='entropy'); ax.plot(p,4*p*(1-p),color=GREEN,lw=1.6,ls='--',label='4 × Gini')
ax.set_xlabel('$p$'); ax.set_ylim(0,1.05); ax.set_xlim(0,1); ax.legend(loc='lower center',fontsize=9,borderaxespad=0.2); grid(ax); sv(fig,'p_24B1.png')

# 24B.2 min_samples_leaf on breast cancer
msl=[1,5,10,20,40]; rows=[]
for m_ in msl:
    t=DTC(min_samples_leaf=m_,random_state=0).fit(Xtr,ytr); rows.append((m_,int(t.get_n_leaves()),int(t.get_depth()),round(float(t.score(Xtr,ytr)),4),round(float(t.score(Xte,yte)),4)))
A['24B.2']=rows
fig,ax=small(); ax.plot(msl,[r[1] for r in rows],'-o',color=PUR,ms=4,lw=1.4); ax.set_xlabel('min_samples_leaf'); ax.set_ylabel('leaves'); ax.set_xscale('log'); ax.set_xticks(msl); ax.set_xticklabels([str(v) for v in msl],fontsize=8.5); grid(ax); sv(fig,'p_24B2.png')

# 24B.3 irrelevant columns
rng=np.random.default_rng(0); Xn=np.c_[Xtr,rng.normal(size=(len(Xtr),5))]; Xne=np.c_[Xte,rng.normal(size=(len(Xte),5))]
t=DTC(random_state=0).fit(Xn,ytr); imp=t.feature_importances_; noise=float(imp[30:].sum())
t0=DTC(random_state=0).fit(Xtr,ytr)
A['24B.3']={'noise_total':round(noise,4),'noise_each':[round(float(v),4) for v in imp[30:]],'test_with':round(float(t.score(Xne,yte)),4),'test_without':round(float(t0.score(Xte,yte)),4),'leaves_with':int(t.get_n_leaves()),'leaves_without':int(t0.get_n_leaves())}
fig,ax=small(); ax.bar(range(5),imp[30:],color=GREY,width=0.6); ax.set_xticks(range(5)); ax.set_xticklabels([f'n{k+1}' for k in range(5)]); ax.set_ylabel('importance'); ax.set_xlabel('noise column'); grid(ax); sv(fig,'p_24B3.png')

# 24C.1 extrapolation
xe=np.linspace(0,5,40); ye=2*xe+1; m=DTR(random_state=0).fit(xe[:,None],ye); xx=np.linspace(0,8,300)
A['24C.1']={'pred_at_8':round(float(m.predict([[8.0]])[0]),3),'true_at_8':17.0}
fig,ax=small(); ax.plot(xx,2*xx+1,color=GREY,lw=1.0,ls=':'); ax.plot(xx,m.predict(xx[:,None]),color=GREEN,lw=1.7); ax.scatter(xe[::3],ye[::3],s=12,color=BLUE,zorder=5); ax.axvline(5,color=PUR,lw=0.9,ls='--'); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); grid(ax); sv(fig,'p_24C1.png')

# 24C.2 a monotone transformation
xv=np.array([1,2,4,8,16.]); yv=np.array([0,0,1,1,1]); mid=(xv[:-1]+xv[1:])/2; mid_log=(np.log2(xv[:-1])+np.log2(xv[1:]))/2
A['24C.2']={'mid_lin':[float(v) for v in mid],'mid_log2_back':[round(float(2**v),3) for v in mid_log]}
t_lin=DTC(max_depth=1,random_state=0).fit(xv[:,None],yv); t_log=DTC(max_depth=1,random_state=0).fit(np.log2(xv)[:,None],yv)
A['24C.2']['thr_lin']=float(t_lin.tree_.threshold[0]); A['24C.2']['thr_log2']=float(t_log.tree_.threshold[0]); A['24C.2']['thr_log2_back']=round(float(2**t_log.tree_.threshold[0]),4)
fig,ax=small(2.4,1.5)
for yy_,lab,xs_ in [(1,'$x$',xv),(0,'$\\log_2 x$',np.log2(xv))]:
    ax.scatter(xs_,[yy_]*5,s=36,c=[BLUE if v==0 else OR for v in yv],marker='o',edgecolor='white',lw=0.7,zorder=5)
ax.set_yticks([0,1]); ax.set_yticklabels(['$\\log_2 x$','$x$']); ax.set_ylim(-0.6,1.6); ax.set_xlabel('value on the axis'); grid(ax); sv(fig,'p_24C2.png')

# 24C.3 seeds
acc=[DTC(random_state=s).fit(Xtr,ytr).score(Xte,yte) for s in range(10)]; A['24C.3']={'acc':[round(float(a),4) for a in acc],'min':round(min(acc),4),'max':round(max(acc),4),'one_patient':round(1/143,4)}
fig,ax=small(); ax.plot(range(10),acc,'o',color=BLUE,ms=5); ax.axhline(np.mean(acc),color=GREEN,lw=1.0,ls='--'); ax.set_ylim(0.88,0.94); ax.set_xlabel('random_state'); ax.set_ylabel('test accuracy'); ax.set_xticks([0,3,6,9]); grid(ax); sv(fig,'p_24C3.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

# All figures of Chapter 24. Run: python3 fig/figs.py  (PNGs into ch24/, numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier as DTC, DecisionTreeRegressor as DTR
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
plt.rcParams.update({'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.labelsize':9,'legend.fontsize':8.5})
NUM={}; CHK={}
def sv(fig,name,printed_px):
    """save and record the smallest printed font size (pt)"""
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible())
    save(fig,name); from PIL import Image
    with Image.open(name) as im: w=im.size[0]/240
    CHK[name]={'fig_in':round(w,2),'printed_in':round(printed_px/96,2),'min_font_design':round(fs,1),'min_font_printed':round(fs*printed_px/96/w,1)}
LO='#FDEBD0'      # light orange (class 1 region)
CMAP=ListedColormap([LB,LO])
def cls_pts(ax,X,y,s=30,**k):
    for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')):
        m=y==cls; ax.scatter(X[m,0],X[m,1],s=s,color=col,marker=mk,edgecolor='white',lw=0.7,zorder=5,**k)
# ---- the loan applicants (age, income, repaid)
AGE=np.array([35,27,22,48,33,24,52,29,41,58]); INC=np.array([22,40,55,28,62,18,36,26,48,75]); Y=np.array([0,1,0,0,1,0,1,0,1,1])
XL=np.c_[AGE,INC]
loan=DTC(random_state=0).fit(XL,Y)

# ---- generic tree drawing from a fitted tree_
def draw_tree(ax,t,names,fs=8.5,bw=1.0,bh=0.42,dy=1.0,counts=True,labels=('0','1'),maxd=None,bwi=None):
    """left branch = condition true. Leaves coloured by class; returns nothing."""
    leaves=[k for k in range(t.node_count) if t.children_left[k]==-1 and (maxd is None or True)]
    pos={}; cnt=[0]
    def lay(k,d):
        if t.children_left[k]==-1: pos[k]=(cnt[0],-d*dy); cnt[0]+=1
        else:
            lay(t.children_left[k],d+1); lay(t.children_right[k],d+1)
            pos[k]=((pos[t.children_left[k]][0]+pos[t.children_right[k]][0])/2,-d*dy)
    lay(0,0)
    for k in range(t.node_count):
        if t.children_left[k]!=-1:
            for ch in (t.children_left[k],t.children_right[k]):
                ax.plot([pos[k][0],pos[ch][0]],[pos[k][1]-bh/2,pos[ch][1]+bh/2],color=GREY,lw=0.9,zorder=1)
    for k in range(t.node_count):
        x,y=pos[k]; n=int(t.n_node_samples[k]); v=t.value[k,0]; bw_=bw if t.children_left[k]==-1 else (bwi or bw)
        if t.children_left[k]==-1:
            c=int(np.argmax(v)); fc=[LB,LO][c]; ec=[BLUE,OR][c]
            txt=f'{labels[c]}\n'+(f'{int(round(v[0]*n))} | {int(round(v[1]*n))}' if counts else f'n = {n}')
        else:
            fc='white'; ec=GREEN; txt=f'{names[t.feature[k]]}\n≤ {t.threshold[k]:.4g}?'
        ax.add_patch(FancyBboxPatch((x-bw_/2,y-bh/2),bw_,bh,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=1.0,zorder=3))
        ax.text(x,y,txt,ha='center',va='center',fontsize=fs,color=NAVY,zorder=4,linespacing=1.05)
    xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
    ax.set_xlim(min(xs)-bw/2-0.05,max(xs)+bw/2+0.05); ax.set_ylim(min(ys)-bh/2-0.05,max(ys)+bh/2+0.05); ax.axis('off')
    return pos

# ---- 24.0-1 opener: the tree and the rectangles
fig,axs=plt.subplots(1,2,figsize=(5.4,2.35),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; draw_tree(ax,loan.tree_,['age','income'],fs=8.8,bw=0.9,bh=0.5,dy=0.95,labels=('defaulted','repaid'))
ax.text(1.0,-2.0*0.95-0.52,'leaf counts: defaulted | repaid',fontsize=8.5,color=GREY,ha='center',va='top'); ax.set_ylim(-2.0*0.95-0.8,0.3)
ax=axs[1]
ax.add_patch(Rectangle((0,0),32,80,fc=LB,ec='none')); ax.add_patch(Rectangle((32,0),50,24.5,fc=LB,ec='none')); ax.add_patch(Rectangle((32,24.5),50,60,fc=LO,ec='none'))
ax.plot([32,32],[0,80],color=GREEN,lw=1.8); ax.plot([32,82],[24.5,24.5],color=GREEN,lw=1.8)
cls_pts(ax,np.c_[INC,AGE],Y,s=30)
ax.set_xlim(10,82); ax.set_ylim(18,64); ax.set_xlabel('income (thousands)'); ax.set_ylabel('age (years)'); ax.grid(False)
ax.text(11,62,'defaulted',fontsize=8.5,color=BLUE,va='top'); ax.text(81,26,'repaid',fontsize=8.5,color=OR,va='bottom',ha='right')
fig.tight_layout(w_pad=0.8); sv(fig,'f240.png',470)

# ---- 24.1-1 the data, with two new applicants
fig,ax=plt.subplots(figsize=(3.9,2.7),dpi=240)
cls_pts(ax,np.c_[INC,AGE],Y,s=40)
for k,(a,i) in enumerate(zip(AGE,INC)): ax.text(i+1.6,a+0.9,str(k+1),fontsize=8.5,color=GREY,ha='left',va='bottom')
NEW=np.array([[45,30],[60,23]])
ax.scatter(NEW[:,0],NEW[:,1],s=70,marker='*',color=PUR,edgecolor='white',lw=0.6,zorder=6)
ax.text(46.5,30,'new A',fontsize=8.5,color=PUR,va='center'); ax.text(61.5,23,'new B',fontsize=8.5,color=PUR,va='center')
ax.set_xlim(10,82); ax.set_ylim(15,65); ax.set_xlabel('income (thousands)'); ax.set_ylabel('age (years)'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.scatter([],[],s=30,color=BLUE,marker='o',label='defaulted'); ax.scatter([],[],s=30,color=OR,marker='s',label='repaid'); ax.legend(loc='upper left',fontsize=8.5,handletextpad=0.2,borderaxespad=0.2)
sv(fig,'f241.png',360)

# ---- 24.2-1 three stages
fig,axs=plt.subplots(1,3,figsize=(5.4,2.35),dpi=240,sharey=True)
stages=[('0  no question yet',[]),('1  income ≤ 32 ?',[('v',32,15,80)]),('2  then age ≤ 24.5 ?',[('v',32,15,80),('h',24.5,32,82)])]
for k,(ax,(title,lines)) in enumerate(zip(axs,stages)):
    if k==0: ax.add_patch(Rectangle((10,15),72,50,fc='#F2F2F2',ec='none'))
    if k>=1:
        ax.add_patch(Rectangle((10,15),22,50,fc=LB,ec='none')); ax.add_patch(Rectangle((32,15),50,50,fc=LO,ec='none'))
    if k==2: ax.add_patch(Rectangle((32,15),50,9.5,fc=LB,ec='none'))
    for kind,v,a,b in lines:
        (ax.plot([v,v],[a,b],color=GREEN,lw=1.8) if kind=='v' else ax.plot([a,b],[v,v],color=GREEN,lw=1.8))
    cls_pts(ax,np.c_[INC,AGE],Y,s=20)
    ax.set_xlim(10,82); ax.set_ylim(15,65); ax.set_title(title,fontsize=9,color=NAVY); ax.set_xlabel('income',fontsize=9); ax.set_xticks([20,40,60,80])
axs[0].set_ylabel('age'); 
lab={0:[('5 | 5',46,60)],1:[('4 | 0',21,60),('1 | 5',57,60)],2:[('4 | 0',21,60),('0 | 5',57,60),('1 | 0',57,19.5)]}
for k,ax in enumerate(axs):
    for t_,x_,y_ in lab[k]: ax.text(x_,y_,t_,fontsize=8.5,color=NAVY,ha='center',va='center',weight='bold')
fig.tight_layout(w_pad=0.5); sv(fig,'f242.png',470)

# ---- 24.3-1 impurity measures
fig,axs=plt.subplots(1,2,figsize=(5.4,2.4),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; p=np.linspace(0.0005,0.9995,400)
g=2*p*(1-p); e=-(p*np.log2(p)+(1-p)*np.log2(1-p)); m=np.minimum(p,1-p)
ax.plot(p,e,color=PUR,lw=1.8,label='entropy'); ax.plot(p,g,color=GREEN,lw=1.8,ls='--',label='Gini'); ax.plot(p,m,color=RED,lw=1.6,ls=':',label='misclassification')
ax.set_xlabel('share $p$ of class 1 in the node'); ax.set_ylabel('impurity'); ax.set_xlim(0,1); ax.set_ylim(0,1.05); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.text(0.13,0.80,'entropy',color=PUR,fontsize=9,ha='center'); ax.text(0.16,0.385,'Gini',color=GREEN,fontsize=9,ha='center'); ax.text(0.5,0.07,'misclassification',color=RED,fontsize=9,ha='center')
ax=axs[1]; labs=['misclass.','Gini','entropy']; A_=[0.25,0.125,0.1887]; B_=[0.25,0.1667,0.3113]
xx=np.arange(3); ax.bar(xx-0.19,A_,0.36,color=BLUE,label='split A: (3,1) | (1,3)'); ax.bar(xx+0.19,B_,0.36,color=OR,label='split B: (4,2) | (0,2)')
for x_,a_,b_ in zip(xx,A_,B_):
    ax.text(x_-0.19,a_+0.008,f'{a_:.3f}'.lstrip('0'),ha='center',fontsize=8.5,color=NAVY); ax.text(x_+0.19,b_+0.008,f'{b_:.3f}'.lstrip('0'),ha='center',fontsize=8.5,color=NAVY)
ax.set_xticks(xx); ax.set_xticklabels(labs); ax.set_ylabel('impurity decrease'); ax.set_ylim(0,0.46); ax.legend(loc='upper left',fontsize=8.2,borderaxespad=0.1,handlelength=1.2)
ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f243.png',470)

# ---- 24.4-1 a regression split
inc6=np.array([20,30,40,50,60,70.]); lim6=np.array([4,5,6,12,13,14.])
thr=(inc6[:-1]+inc6[1:])/2; sse=lambda v:float(np.sum((v-v.mean())**2))
S=[sse(lim6[inc6<=t])+sse(lim6[inc6>t]) for t in thr]; NUM['reg_sse']=[round(s_,2) for s_ in S]
fig,axs=plt.subplots(1,2,figsize=(5.2,2.35),dpi=240,gridspec_kw={'width_ratios':[1.1,1]})
ax=axs[0]; ax.axvline(45,color=PUR,lw=1.2,ls='--'); ax.plot([15,45],[5,5],color=GREEN,lw=2.2); ax.plot([45,75],[13,13],color=GREEN,lw=2.2)
ax.scatter(inc6,lim6,s=36,color=BLUE,edgecolor='white',lw=0.8,zorder=5)
for x_,y_,m_ in zip(inc6,lim6,[5,5,5,13,13,13]): ax.plot([x_,x_],[y_,m_],color=RED,lw=1.2,zorder=3)
ax.text(46.5,2.0,'split at 45',color=PUR,fontsize=9); ax.text(16,5.9,'mean 5',color=GREEN,fontsize=9); ax.text(74,11.9,'mean 13',color=GREEN,fontsize=9,ha='right',va='top')
ax.set_xlabel('income (thousands)'); ax.set_ylabel('credit limit (thousands)'); ax.set_xlim(14,76); ax.set_ylim(0,16); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax=axs[1]; ax.plot(thr,S,'-o',color=RED,ms=4.5,lw=1.4); k=int(np.argmin(S)); ax.plot(thr[k],S[k],'*',color=GREEN,ms=13,mec='white',zorder=5)
ax.axhline(100,color=GREY,lw=0.8,ls=':'); ax.text(26,96,'SSE of the parent: 100',fontsize=8.5,color=GREY,va='top')
ax.text(thr[k]+1.5,S[k]+4,f'minimum: {S[k]:.0f}',fontsize=9,color=GREEN,ha='left'); ax.set_xlabel('threshold'); ax.set_ylabel('SSE after the split'); ax.set_ylim(0,112); ax.set_xlim(20,70); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); sv(fig,'f244.png',450)

# ---- 24.4-2 regression trees of growing depth
rng=np.random.default_rng(2); xs=np.sort(rng.uniform(0,6,45)); ys=np.sin(xs)+rng.normal(0,0.3,45); xg=np.linspace(0,6,500)[:,None]
fig,axs=plt.subplots(1,4,figsize=(5.4,1.75),dpi=240,sharey=True); NUM['sine_trees']={}
for ax,d in zip(axs,[1,2,4,None]):
    m=DTR(max_depth=d,random_state=0).fit(xs[:,None],ys); ax.plot(xg[:,0],np.sin(xg[:,0]),color=GREY,lw=0.9,ls=':'); ax.plot(xg[:,0],m.predict(xg),color=GREEN,lw=1.6)
    ax.scatter(xs,ys,s=7,color=BLUE,lw=0,zorder=4); ax.set_title(f'depth {d}' if d else 'no limit',fontsize=9,color=NAVY); ax.set_xticks([0,3,6]); ax.set_xlabel('$x$',fontsize=9)
    NUM['sine_trees'][str(d)]={'leaves':int(m.get_n_leaves()),'train_r2':round(float(m.score(xs[:,None],ys)),3)}
axs[0].set_ylabel('$y$'); axs[0].set_yticks([-1,0,1]); fig.tight_layout(w_pad=0.4); sv(fig,'f245.png',470)

# ---- 24.5-1 the pruning sequence of the loan tree
path=DTC(random_state=0).cost_complexity_pruning_path(XL,Y); al=path.ccp_alphas; NUM['loan_path']={'alphas':[round(float(a),4) for a in al],'impurities':[round(float(a),4) for a in path.impurities]}
fig,axs=plt.subplots(1,3,figsize=(5.4,2.0),dpi=240)
for k,(ax,a) in enumerate(zip(axs,al)):
    m=DTC(random_state=0,ccp_alpha=a if k==0 else a).fit(XL,Y); t=m.tree_
    pos=draw_tree(ax,t,['age','income'],fs=8.4,bw=0.9,bh=0.5,dy=0.95,labels=('defaulted','repaid'))
    xs_=[p[0] for p in pos.values()]; cx=(min(xs_)+max(xs_))/2
    ax.set_xlim(cx-1.55,cx+1.55); ax.set_ylim(-2*0.95-1.05,0.4)
    ax.text(cx,-2*0.95-0.5,f'{m.get_n_leaves()} '+('leaf' if m.get_n_leaves()==1 else 'leaves')+f',  $R(T)$ = {path.impurities[k]:.3f}',fontsize=8.8,color=NAVY,ha='center',va='top'); ax.text(cx,-2*0.95-0.9,[f'best for $\\alpha$ < {al[1]:.3f}',f'best for {al[1]:.3f} ≤ $\\alpha$ < {al[2]:.3f}',f'best for $\\alpha$ ≥ {al[2]:.3f}'][k],fontsize=8.8,color=PUR,ha='center',va='top')
fig.tight_layout(w_pad=0.3); sv(fig,'f246.png',470)

# ---- 24.7-1 flow chart of fit
fig,ax=new(3.8,5.8)
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.3): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
A=lambda p,q,**k: arr(ax,p,q,c=NAVY,lw=0.9,ms=8,**k)
def Ln(*pts): ax.plot([p[0] for p in pts],[p[1] for p in pts],color=NAVY,lw=0.9)
def dia(xc,yc,w,h,t,fs=8.2): ax.add_patch(Polygon([(xc,yc+h/2),(xc+w/2,yc),(xc,yc-h/2),(xc-w/2,yc)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(xc,yc,t,ha='center',va='center',fontsize=fs,color=NAVY,linespacing=1.1)
def yn(x,y,t,ha='left'): ax.text(x,y,t,fontsize=8,color=GREY,style='italic',ha=ha,va='center')
X0=2.7
fb(X0,9.1,4.6,0.55,'Input:  $\\mathbf{X}$ (NaN allowed),  $y$,  weights',fc=LGREY,ec=GREY)
fb(X0,8.3,4.6,0.55,'Put the root (all samples) on the list')
dia(X0,7.1,3.2,1.1,'list empty?')
dia(X0,5.5,3.4,1.5,'stop rule met?\npure, max_depth,\nsample-size limits')
fb(X0,3.95,4.6,1.2,'For each of max_features features:\nsort, scan the midpoints,\nkeep the best (feature, threshold)')
dia(X0,2.3,4.4,1.3,'decrease ≥\nmin_impurity_decrease?')
fb(X0,0.9,4.6,0.7,'Split: $x_j \\leq t$ goes left, the rest right;\nboth children join the list')
fb(5.95,3.95,1.65,1.1,'Leaf:\nstore class\nfractions',fc='#E3F4EA',ec=GREEN)
fb(X0,-0.5,4.6,0.7,'If ccp_alpha > 0: prune (§24.5)\nStore tree_ ;  return self',fc=LGREY,ec=GREY)
A((X0,8.825),(X0,8.575)); A((X0,8.025),(X0,7.65)); A((X0,6.55),(X0,6.25)); A((X0,4.75),(X0,4.55)); A((X0,3.35),(X0,2.95)); A((X0,1.65),(X0,1.25))
Ln((X0-1.6,7.1),(0.15,7.1),(0.15,-0.5)); A((0.15,-0.5),(0.4,-0.5))
Ln((X0+1.7,5.5),(5.95,5.5)); A((5.95,5.5),(5.95,4.5)); Ln((X0+2.2,2.3),(5.95,2.3)); A((5.95,2.3),(5.95,3.4))
Ln((6.775,3.95),(6.85,3.95),(6.85,7.85),(X0,7.85)); Ln((X0+2.3,0.9),(6.85,0.9),(6.85,3.95))
yn(0.3,7.3,'yes'); yn(X0+0.12,6.4,'no'); yn(4.6,5.7,'yes'); yn(X0+0.12,4.65,'no'); yn(5.0,2.5,'no'); yn(X0+0.12,1.45,'yes')
ax.set_xlim(0,6.95); ax.set_ylim(-1.0,9.5); sv(fig,'f247.png',360)

# ---- 24.10-1 depth gallery on the moons
Xm,ym=make_moons(n_samples=300,noise=0.3,random_state=0); Xmt,Xme,ymt,yme=train_test_split(Xm,ym,test_size=0.33,random_state=0,stratify=ym)
gx,gy=np.meshgrid(np.linspace(-1.7,2.7,300),np.linspace(-1.2,1.7,300)); NUM['gallery']={}
fig,axs=plt.subplots(1,4,figsize=(5.2,1.95),dpi=240,sharey=True)
for ax,d in zip(axs,[1,2,4,None]):
    m=DTC(max_depth=d,random_state=0).fit(Xmt,ymt); Z=m.predict(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx,gy,Z,levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx,gy,Z,levels=[0.5],colors=[GREEN],linewidths=1.3)
    cls_pts(ax,Xmt,ymt,s=7); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f'depth {d}' if d else 'no limit',fontsize=9,color=NAVY)
    acc=m.score(Xme,yme); ax.text(0.97,0.04,f'test {acc:.2f}',transform=ax.transAxes,fontsize=8.5,color=NAVY,ha='right',va='bottom',bbox=dict(fc='white',ec='none',pad=1.2,alpha=0.85))
    NUM['gallery'][str(d)]={'leaves':int(m.get_n_leaves()),'train':round(float(m.score(Xmt,ymt)),4),'test':round(float(acc),4)}
fig.tight_layout(w_pad=0.3); sv(fig,'f2410a.png',470)

# ---- 24.10-2 validation curves (numbers from code/choose_depth.py)
V=json.load(open('verify/choose_depth.json'))
fig,axs=plt.subplots(1,2,figsize=(5.2,2.3),dpi=240)
for ax,key,lab in zip(axs,['max_depth','min_samples_leaf'],['max_depth','min_samples_leaf']):
    g=np.array(V[key]['grid']); tr=np.array(V[key]['train']); va=np.array(V[key]['valid']); sd=np.array(V[key]['sd']); k=int(np.argmax(va))
    ax.plot(g,tr,'-o',color=BLUE,ms=3.2,lw=1.4,label='training folds'); ax.plot(g,va,'-s',color=GREY,ms=3.2,lw=1.4,label='validation folds')
    ax.fill_between(g,va-sd,va+sd,color=GREY,alpha=0.18,lw=0); ax.axvline(g[k],color=PUR,lw=1.0,ls='--')
    ax.set_xlabel(lab); ax.set_ylim(0.75,1.02); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
    if key=='max_depth': ax.set_xticks([1,3,6,9,12])
    if key=='min_samples_leaf': ax.set_xscale('log'); ax.set_xticks([1,3,10,30]); ax.set_xticklabels(['1','3','10','30'])
axs[0].set_ylabel('accuracy'); axs[0].legend(loc='lower right',fontsize=8.3,borderaxespad=0.2,handlelength=1.8)
fig.tight_layout(w_pad=1.0); sv(fig,'f2410b.png',450)

# ---- 24.10-3 instability: remove one point
rs=np.random.RandomState(5); Xi,yi=make_moons(n_samples=60,noise=0.3,random_state=1)
gx2,gy2=np.meshgrid(np.linspace(-1.7,2.7,260),np.linspace(-1.2,1.7,260)); G=np.c_[gx2.ravel(),gy2.ravel()]
base=DTC(random_state=0).fit(Xi,yi); Zb=base.predict(G)
near=(np.sqrt(((G[:,None,:]-Xi[None,:,:])**2).sum(-1)).min(1)<0.4)      # the region where the data live
diff=[]
for k in range(len(yi)):
    m=DTC(random_state=0).fit(np.delete(Xi,k,0),np.delete(yi,k)); diff.append(float(np.mean((m.predict(G)!=Zb)[near])))
diff=np.array(diff); kmax=int(np.argmax(diff)); NUM['instab']={'near_share':round(float(near.mean()),3),'median':round(float(np.median(diff)),4),'max':round(float(diff.max()),4),'min':round(float(diff.min()),4),'n':len(yi),'frac_zero':round(float(np.mean(diff==0)),3),'leaves':int(base.get_n_leaves())}
rr=np.random.default_rng(0); d10=[]
for _ in range(100):
    drop=rr.choice(len(yi),6,replace=False); m=DTC(random_state=0).fit(np.delete(Xi,drop,0),np.delete(yi,drop)); d10.append(float(np.mean((m.predict(G)!=Zb)[near])))
NUM['instab']['drop6_median']=round(float(np.median(d10)),4); NUM['instab']['drop6_min']=round(float(np.min(d10)),4); NUM['instab']['drop6_max']=round(float(np.max(d10)),4); NUM['instab']['n_removals_unchanged']=int(np.sum(diff==0))
m2=DTC(random_state=0).fit(np.delete(Xi,kmax,0),np.delete(yi,kmax)); Z2=m2.predict(G)
fig,axs=plt.subplots(1,3,figsize=(5.4,1.95),dpi=240,sharey=True)
for ax,(Z,ttl,pts) in zip(axs,[(Zb,'all 60 points',None),(Z2,'one point removed',kmax),(None,'where they differ',kmax)]):
    if Z is not None:
        ax.contourf(gx2,gy2,Z.reshape(gx2.shape),levels=[-0.5,0.5,1.5],colors=[LB,LO]); ax.contour(gx2,gy2,Z.reshape(gx2.shape),levels=[0.5],colors=[GREEN],linewidths=1.3)
    else:
        ax.contourf(gx2,gy2,(Z2!=Zb).reshape(gx2.shape).astype(float),levels=[-0.5,0.5,1.5],colors=['white',PINK]); ax.contour(gx2,gy2,Zb.reshape(gx2.shape),levels=[0.5],colors=[GREEN],linewidths=0.9,linestyles='--'); ax.contour(gx2,gy2,Z2.reshape(gx2.shape),levels=[0.5],colors=[GREEN],linewidths=1.3)
    msk=np.ones(len(yi),bool)
    cls_pts(ax,Xi[msk],yi[msk],s=10)
    if pts is not None: ax.scatter(*Xi[pts],s=70,facecolors='none',edgecolors=RED,lw=1.4,zorder=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl,fontsize=9,color=NAVY)
axs[2].text(0.03,0.97,f'{100*diff[kmax]:.0f}% of the\ndata region',transform=axs[2].transAxes,fontsize=8.5,color=RED,ha='left',va='top')
fig.tight_layout(w_pad=0.3); sv(fig,'f2410c.png',470)

# ---- 24.11-1 exclusive-or: no gain at the root, a large gain one step later
rng=np.random.default_rng(3); Xx=rng.uniform(-1,1,(240,2)); yx=((Xx[:,0]>0)!=(Xx[:,1]>0)).astype(int); flip=rng.random(240)<0.06; yx=np.where(flip,1-yx,yx)
def gini_(y): 
    if len(y)==0: return 0.0
    p=y.mean(); return 2*p*(1-p)
def gain(idx,j,t,N=len(yx)):
    S=idx; L=S[Xx[S,j]<=t]; R=S[Xx[S,j]>t]
    return len(S)/N*(gini_(yx[S])-len(L)/len(S)*gini_(yx[L])-len(R)/len(S)*gini_(yx[R]))
allidx=np.arange(len(yx)); left=allidx[Xx[:,0]<=0]; ts=np.linspace(-0.9,0.9,181)
g1=[gain(allidx,0,t) for t in ts]; g2=[gain(allidx,1,t) for t in ts]; g3=[gain(left,1,t) for t in ts]
NUM['xor']={'root_max_x1':round(max(g1),4),'root_max_x2':round(max(g2),4),'child_max':round(max(g3),4),'root_gini':round(gini_(yx),4)}
fig,axs=plt.subplots(1,2,figsize=(5.2,2.35),dpi=240,gridspec_kw={'width_ratios':[1,1.15]})
ax=axs[0]; cls_pts(ax,Xx,yx,s=9); ax.axvline(0,color=GREEN,lw=1.4); ax.plot([-1,0],[0,0],color=GREEN,lw=1.4,ls='--'); ax.plot([0,1],[0,0],color=GREEN,lw=1.4,ls='--')
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1]); ax.set_aspect('equal')
ax=axs[1]; ax.plot(ts,g1,color=BLUE,lw=1.5,label='root, split on $x_1$'); ax.plot(ts,g2,color=OR,lw=1.5,ls='--',label='root, split on $x_2$'); ax.plot(ts,g3,color=GREEN,lw=1.8,label='in $x_1 \\leq 0$, split on $x_2$')
ax.set_xlabel('threshold $t$'); ax.set_ylabel('impurity decrease'); ax.set_ylim(-0.01,0.31); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(loc='upper left',fontsize=8.2,borderaxespad=0.1,handlelength=1.8)
fig.tight_layout(w_pad=1.0); sv(fig,'f2411.png',450)

# ---- 24.12-1 the pruning path of the breast-cancer tree
cd=load_breast_cancer(); Xt,Xe,yt,ye=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
pth=DTC(random_state=0).cost_complexity_pruning_path(Xt,yt); als=pth.ccp_alphas[:-1]; R_=pth.impurities[:-1]
tr_,cv_,te_,lv_=[],[],[],[]
for a in als:
    m=DTC(ccp_alpha=a,random_state=0); cv_.append(cross_val_score(m,Xt,yt,cv=5).mean()); m.fit(Xt,yt); tr_.append(m.score(Xt,yt)); te_.append(m.score(Xe,ye)); lv_.append(m.get_n_leaves())
kb=int(np.argmax(cv_)); NUM['path']={'n_alphas':len(pth.ccp_alphas),'best_alpha':round(float(als[kb]),4),'best_cv':round(float(cv_[kb]),4),'leaves':[int(v) for v in lv_],'alphas':[round(float(a),4) for a in als],'cv':[round(float(v),4) for v in cv_],'test':[round(float(v),4) for v in te_]}
fig,axs=plt.subplots(1,2,figsize=(5.2,2.35),dpi=240)
ax=axs[0]; ax.step(als,R_,where='post',color=RED,lw=1.6); ax.plot(als,R_,'o',color=RED,ms=3)
ax.set_xlabel('$\\alpha$'); ax.set_ylabel('$R(T_\\alpha)$',color=RED); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.set_xlim(-0.001,0.028)
a2=ax.twinx(); a2.step(als,lv_,where='post',color=PUR,lw=1.4,ls='--'); a2.set_ylabel('leaves',color=PUR); a2.spines['right'].set_visible(True); a2.set_ylim(0,19)
ax=axs[1]; ax.plot(als,tr_,'-o',color=BLUE,ms=3,lw=1.3,label='training'); ax.plot(als,cv_,'-s',color=GREY,ms=3,lw=1.3,label='5-fold validation'); ax.plot(als,te_,':^',color=OR,ms=3.4,lw=1.3,label='test')
ax.axvline(als[kb],color=PUR,lw=1.0,ls='--'); ax.set_xlabel('$\\alpha$'); ax.set_ylabel('accuracy'); ax.set_xlim(-0.001,0.028); ax.set_ylim(0.80,1.01); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.legend(loc='lower left',fontsize=8.2,borderaxespad=0.2,handlelength=1.8)
fig.tight_layout(w_pad=1.2); sv(fig,'f2412a.png',450)

# ---- 24.12-2 the pruned tree
pr=DTC(ccp_alpha=als[kb],random_state=0).fit(Xt,yt); nm=list(cd.feature_names)
fig,ax=plt.subplots(figsize=(5.4,2.5),dpi=240); pos=draw_tree(ax,pr.tree_,nm,fs=8.5,bw=0.95,bwi=1.35,bh=0.5,dy=1.0,labels=('malignant','benign'))
for k,(x_,y_) in pos.items():
    pass
sv(fig,'f2412b.png',470)
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); json.dump(CHK,open('verify/fig_textsize.json','w'),indent=1)
bad={k:v for k,v in CHK.items() if v['min_font_printed']<7.5}; print('figures below 7.5 pt:',bad); print(json.dumps(NUM))

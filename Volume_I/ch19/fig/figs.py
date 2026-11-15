# All figures of Chapter 19. Run: python3 fig/figs.py  (writes PNGs into ch19/ and numbers into verify/fig_numbers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('ignore')
from figstyle import *
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
NUM={}
ORL='#FCE3BE'
cm2=LinearSegmentedColormap.from_list('bo',[LB,'#FFFFFF',ORL])
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ---- 19.0-1 opener: two clouds, axis-aligned ellipses, boundary
rng=np.random.default_rng(4)
A=rng.normal([2.2,3.0],[0.8,0.8],(26,2)); Bp=rng.normal([6.0,6.6],[1.7,1.5],(26,2))
Xo=np.r_[A,Bp]; yo=np.r_[np.zeros(26),np.ones(26)]; g=GaussianNB().fit(Xo,yo)
fig,ax=plt.subplots(figsize=(4.7,1.75),dpi=240)
gx,gy=np.meshgrid(np.linspace(-0.5,10.2,300),np.linspace(0.2,10.4,300)); P=g.predict_proba(np.c_[gx.ravel(),gy.ravel()])[:,1].reshape(gx.shape)
ax.contourf(gx,gy,P,levels=np.linspace(0,1,21),cmap=cm2,zorder=0); ax.contour(gx,gy,P,levels=[0.5],colors=[GREEN],linewidths=2.0,zorder=3)
for k,(c,mk) in enumerate([(BLUE,'o'),(OR,'^')]):
    Xk=Xo[yo==k]; ax.scatter(Xk[:,0],Xk[:,1],s=24,color=c,marker=mk,edgecolor='white',lw=0.6,zorder=5)
    for s_ in (1,2): ax.add_patch(Ellipse(g.theta_[k],2*s_*np.sqrt(g.var_[k,0]),2*s_*np.sqrt(g.var_[k,1]),fill=False,ec=GREEN,lw=0.9,ls='-' if s_==1 else (0,(3,2)),zorder=4))
ax.text(0.0,1.2,'class 0',color=BLUE,fontsize=9.5,weight='bold'); ax.text(8.0,9.4,'class 1',color=OR,fontsize=9.5,weight='bold')
ax.text(5.1,1.0,'boundary  $P=0.5$',color=GREEN,fontsize=9.5,weight='bold',va='center')
ax.text(-0.3,10.2,'ellipses: 1 and 2 standard\ndeviations of each feature',fontsize=8.5,color=GREEN,ha='left',va='top')
ax.set_xlim(-0.5,10.2); ax.set_ylim(0.2,10.4); ax.set_xlabel('feature $x_1$'); ax.set_ylabel('feature $x_2$'); ax.set_xticks([]); ax.set_yticks([]); save(fig,'f1900.png')

# ---- 19.2-1 graphical model + parameter count
fig,axs=plt.subplots(1,2,figsize=(5.4,2.45),dpi=240,gridspec_kw={'width_ratios':[1,1.25]})
ax=axs[0]; ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(0,4); ax.set_ylim(-0.9,3.4)
ax.add_patch(Circle((2,2.8),0.36,fc='#FDF0E0',ec=OR,lw=1.2)); ax.text(2,2.8,'$k$',ha='center',va='center',fontsize=12,color=NAVY)
for j,xx_ in enumerate([0.55,1.5,3.45]):
    ax.add_patch(Circle((xx_,0.85),0.36,fc=LB,ec=BLUE,lw=1.2)); lab=['$x_1$','$x_2$','$x_p$'][j]; ax.text(xx_,0.85,lab,ha='center',va='center',fontsize=11,color=NAVY)
    arr(ax,(2+(xx_-2)*0.17,2.45),(xx_+(2-xx_)*0.12,1.2),c=GREEN,lw=1.1,ms=9)
ax.text(2.5,0.85,'. . .',ha='center',va='center',fontsize=12,color=GREY)
ax.text(2,0.1,'given the class, the features\nare independent',ha='center',va='top',fontsize=8.8,color=NAVY)
ax.text(2.85,2.8,'class',fontsize=8.8,color=OR,va='center'); ax.set_title('The naive structure',fontsize=9.5,color=NAVY)
ax=axs[1]; p=np.arange(1,21); ax.semilogy(p,2*(2.0**p-1)+1,color=RED,lw=1.8,marker='s',ms=3.2); ax.semilogy(p,2*p+1,color=GREEN,lw=1.8,marker='o',ms=3.2)
ax.text(2.2,3e4,'full joint table\n$2(2^{p}-1)+1$',color=RED,fontsize=8.8,ha='left',va='center'); ax.text(9.2,2*9.2+1+34,'naive Bayes: $2p+1$',color=GREEN,fontsize=8.8,ha='left',va='bottom')
ax.set_xlabel('number of binary features $p$'); ax.set_ylabel('numbers to estimate (2 classes)'); ax.set_title('What has to be learned',fontsize=9.5,color=NAVY); grid(ax); ax.set_xlim(0,20.5)
fig.tight_layout(w_pad=1.0); save(fig,'f1921.png')

# ---- 19.6-1 flow chart
fig,ax=new(4.5,4.6); cx=3.78
def fb(xc,yc,w,h,t,fc=LB,ec=BLUE,fs=8.5): box(ax,xc-w/2,yc-h/2,w,h,t,fc=fc,ec=ec,fs=fs,r=0.06)
Aw=lambda p,q: arr(ax,p,q,c=NAVY,lw=0.9,ms=8)
Ln=lambda xs,ys: ax.plot(xs,ys,color=NAVY,lw=0.9,solid_capstyle='butt')
fb(cx,8.1,7.2,0.55,'FIT:  group by class; count $n_k$;  $P(k)=n_k/n$')
ax.add_patch(Polygon([(cx,7.65),(cx+1.4,7.15),(cx,6.65),(cx-1.4,7.15)],fc='#FDF0E0',ec=OR,lw=0.9)); ax.text(cx,7.15,'feature type?',ha='center',va='center',fontsize=8.5,color=NAVY)
xs=[cx-2.85,cx-0.95,cx+0.95,cx+2.85]; ws=[1.8,1.8,1.8,1.6]
txt=['real numbers\nGaussianNB:\nmean, variance\n(Eq. 19.4-2)','counts\nMultinomialNB,\nComplementNB:\n$N_{kj}+\\alpha$','present / absent\nBernoulliNB:\n$N_{kj}+\\alpha$ after\nbinarising','categories\nCategoricalNB:\ncounts per\ncategory']
for xc,w,t in zip(xs,ws,txt):
    fb(xc,5.6,w,1.5,t,fc='#E3F4EA',ec=GREEN,fs=8.2); Aw((xc,6.5),(xc,6.36)); Ln([xc,xc],[4.85,4.55])
Aw((cx,7.82),(cx,7.65)); Aw((cx,6.65),(cx,6.5)); Ln([xs[0],xs[3]],[6.5,6.5]); Ln([xs[0],xs[3]],[4.55,4.55])
fb(cx,3.95,7.2,0.7,'Store log-priors and log-likelihood parameters\n(class_log_prior_, feature_log_prob_, theta_, var_)',fc=LGREY,ec=GREY,fs=8.3)
Aw((cx,4.55),(cx,4.3))
fb(cx,2.75,7.2,0.95,'PREDICT:  score of every class  $s_k=\\ln P(k)+\\sum_j\\ln p(x_j\\mid k)$\n(Eqs. 19.3-3, 19.4-3, 19.4-6, 19.4-8, 19.4-9)',fs=8.3)
fb(cx,1.45,7.2,0.95,'Subtract $\\ln\\sum_l e^{s_l}$  (log-sum-exp):\npredict_log_proba, predict_proba',fs=8.3)
fb(cx,0.4,7.2,0.55,'predict:  the class with the largest $s_k$',fc='#E3F4EA',ec=GREEN,fs=8.5)
Aw((cx,3.6),(cx,3.23)); Aw((cx,2.27),(cx,1.95)); Aw((cx,0.97),(cx,0.68))
ax.set_xlim(0.0,7.56); ax.set_ylim(0.05,8.45); save(fig,'f1961.png')

# ---- 19.9-1 smoothing gallery on the spam corpus
words=['free','lunch','meeting','now','prize','report','win']
N=np.array([[0,2,3,1,0,2,0],[3,0,0,3,2,0,2]],float)
fig,axs=plt.subplots(2,2,figsize=(4.6,3.0),dpi=240,sharex=True,sharey=True); NUM['gallery']={}
for ax,a,t in zip(axs.ravel(),[0,0.1,1,10],['$\\alpha=0$  (no smoothing)','$\\alpha=0.1$','$\\alpha=1$  (the default)','$\\alpha=10$']):
    th=(N+a)/(N+a).sum(1,keepdims=True); NUM['gallery'][str(a)]={'ham':[round(float(v),4) for v in th[0]],'spam':[round(float(v),4) for v in th[1]]}
    yy=np.arange(7)[::-1]
    ax.barh(yy+0.2,th[0],height=0.4,color=BLUE,label='ham'); ax.barh(yy-0.2,th[1],height=0.4,color=OR,hatch='////',edgecolor='white',lw=0,label='spam')
    ax.axvline(1/7,color=GREY,lw=0.8,ls=':'); ax.set_yticks(yy); ax.set_yticklabels(words,fontsize=8.5); ax.set_title(t,fontsize=9,color=PUR,pad=2); ax.set_xlim(0,0.44); ax.tick_params(labelsize=8.5)
    ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
    if a==0:
        for r_,k_ in [(0,0),(1,0)]:
            for j_ in np.where(N[r_]==0)[0]: ax.text(0.008,yy[j_]+(0.2 if r_==0 else -0.2),'0',fontsize=8,color=RED,va='center')
axs[0,0].legend(fontsize=8.5,loc='lower right',handlelength=1.2,borderaxespad=0.2)
for ax in axs[1]: ax.set_xlabel('$\\theta_{kj}$  =  $P$(word | class)',fontsize=8.8)
fig.tight_layout(h_pad=0.4,w_pad=0.5); save(fig,'f1991.png')

# ---- 19.9-2 validation curves
d=load_digits(); Xtr,Xte,ytr,yte=train_test_split(d.data,d.target,test_size=0.25,random_state=0,stratify=d.target)
from sklearn.model_selection import validation_curve
vs=np.logspace(-9,0,10); tr,va=validation_curve(GaussianNB(),Xtr,ytr,param_name='var_smoothing',param_range=vs,cv=5)
NUM['vs']={'vs':list(map(float,vs)),'train':[round(float(v),4) for v in tr.mean(1)],'cv':[round(float(v),4) for v in va.mean(1)]}
al=np.logspace(-4,2,7); cvs=ShuffleSplit(20,train_size=50,test_size=500,random_state=0); accs=np.array([cross_val_score(MultinomialNB(alpha=a),d.data,d.target,cv=cvs) for a in al])
NUM['alpha_full_test']={str(a):round(float(MultinomialNB(alpha=a).fit(Xtr,ytr).score(Xte,yte)),4) for a in [1e-3,1e-2,0.1,1,10]}
NUM['alpha50']={'alpha':list(map(float,al)),'mean':[round(float(v),4) for v in accs.mean(1)],'sd':[round(float(v),4) for v in accs.std(1)]}
fig,axs=plt.subplots(1,2,figsize=(5.4,2.45),dpi=240)
ax=axs[0]; ax.semilogx(vs,tr.mean(1),'-s',color=BLUE,ms=3.5,lw=1.5,label='training'); ax.semilogx(vs,va.mean(1),'--o',color=GREY,ms=3.5,lw=1.5,label='validation (5-fold)')
ax.fill_between(vs,va.mean(1)-va.std(1),va.mean(1)+va.std(1),color=GREY,alpha=0.18,lw=0); ax.axvline(1e-9,color=PUR,lw=1,ls=(0,(4,2))); ax.text(2e-9,0.57,'default',color=PUR,fontsize=8.5,rotation=90,va='bottom',ha='left')
ax.set_ylim(0.55,1.0); ax.set_xlabel('var_smoothing',color=PUR); ax.set_ylabel('accuracy'); ax.set_title('GaussianNB on digits',fontsize=9.5,color=NAVY); ax.legend(fontsize=8.2,loc='lower right'); grid(ax); ax.set_xticks([1e-9,1e-6,1e-3,1])
ax=axs[1]; ax.semilogx(al,accs.mean(1),'--o',color=GREY,ms=3.5,lw=1.5); ax.fill_between(al,accs.mean(1)-accs.std(1),accs.mean(1)+accs.std(1),color=GREY,alpha=0.18,lw=0)
ax.axvline(1,color=PUR,lw=1,ls=(0,(4,2))); ax.text(1.25,0.46,'default',color=PUR,fontsize=8.5,rotation=90,va='bottom',ha='left')
ax.set_xlabel('alpha',color=PUR); ax.set_title('MultinomialNB, 50 training images',fontsize=9.5,color=NAVY); ax.set_ylim(0.4,0.95); grid(ax); ax.set_xticks([1e-4,1e-2,1,100])
fig.tight_layout(w_pad=1.0); save(fig,'f1992.png')

# ---- 19.10-1 counting evidence twice; reliability
fig,axs=plt.subplots(1,2,figsize=(5.4,2.5),dpi=240)
ax=axs[0]; cps=np.arange(1,11); lo=0.3; sig=lambda z:1/(1+np.exp(-z))
ax.plot(cps,sig(lo*cps),'-o',color=PUR,ms=4,lw=1.6); ax.axhline(sig(lo),color=NAVY,lw=1.1,ls='--')
ax.text(10,sig(lo)-0.02,'one copy (the true evidence)',fontsize=8.3,color=NAVY,ha='right',va='top'); ax.text(5.6,0.66,'naive Bayes with the\nsame column repeated',fontsize=8.3,color=PUR,ha='left',va='center')
ax.set_xlabel('number of copies of the feature'); ax.set_ylabel('$P$(class 1 | $\\mathbf{x}$)'); ax.set_ylim(0.45,1.0); ax.set_xticks([1,2,4,6,8,10]); grid(ax); ax.set_title('Evidence counted twice',fontsize=9.5,color=NAVY)
NUM['copies']={'c':[int(c) for c in cps],'post':[round(float(sig(lo*c)),4) for c in cps],'true':round(float(sig(lo)),4)}
ax=axs[1]; from sklearn.linear_model import LogisticRegression
mods_=[('Gaussian',GaussianNB()),('Gaussian\ntuned',GaussianNB(var_smoothing=0.1)),('Multi-\nnomial',MultinomialNB()),('logistic\nregr.',LogisticRegression(max_iter=2000))]; NUM['reliab']={}
cf=[];ac=[]
for nm,m in mods_:
    m.fit(Xtr,ytr); pr=m.predict_proba(Xte); cf.append(pr.max(1).mean()); ac.append(m.score(Xte,yte)); NUM['reliab'][nm.replace('\n',' ')]={'conf':round(float(cf[-1]),4),'acc':round(float(ac[-1]),4)}
xx=np.arange(4); ax.bar(xx-0.2,cf,width=0.38,color=MAG,hatch='////',edgecolor='white',lw=0,label='mean predicted probability\nof the winning class'); ax.bar(xx+0.2,ac,width=0.38,color=GREY,label='accuracy')
for x_,v in zip(xx-0.2,cf): ax.text(x_,v+0.01,f'{v:.2f}',ha='center',fontsize=8.2,color=MAG)
for x_,v in zip(xx+0.2,ac): ax.text(x_,v+0.01,f'{v:.2f}',ha='center',fontsize=8.2,color=GREY)
ax.set_xticks(xx); ax.set_xticklabels([n for n,_ in mods_],fontsize=8.2); ax.set_ylim(0.5,1.45); ax.set_yticks([0.5,0.75,1.0]); ax.legend(fontsize=8.4,loc='upper center',ncol=1,handlelength=1.2,bbox_to_anchor=(0.5,1.03)); ax.set_title('Digits: confidence against accuracy',fontsize=9.5,color=NAVY,pad=3); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.0); save(fig,'f19101.png')

# ---- 19.11-1 case study
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB, BernoulliNB, CategoricalNB
from sklearn.dummy import DummyClassifier
gs=GridSearchCV(GaussianNB(),{'var_smoothing':np.logspace(-9,0,10)},cv=5).fit(Xtr,ytr)
mods=[('baseline',DummyClassifier()),('Gaussian',GaussianNB()),('Gaussian, tuned',gs.best_estimator_),('Multinomial',MultinomialNB()),('Complement',ComplementNB()),('Bernoulli',BernoulliNB(binarize=8)),('Categorical',CategoricalNB(min_categories=17))]
acc=[m.fit(Xtr,ytr).score(Xte,yte) for _,m in mods]; NUM['case_acc']={n:round(float(a),4) for (n,_),a in zip(mods,acc)}
fig=plt.figure(figsize=(4.7,2.2),dpi=240); gs_=fig.add_gridspec(1,2,width_ratios=[1.0,1.1],wspace=0.08)
ax=fig.add_subplot(gs_[0]); yy=np.arange(len(mods))[::-1]; ax.barh(yy,acc,color=[LGREY]+[GREY]*6,height=0.62); 
for y_,a in zip(yy,acc): ax.text(a+0.012,y_,f'{a:.3f}',va='center',fontsize=8.2,color=NAVY)
ax.set_yticks(yy); ax.set_yticklabels([n for n,_ in mods],fontsize=8.5); ax.set_xlim(0,1.15); ax.set_xticks([0,0.5,1.0]); ax.set_xlabel('test accuracy'); ax.set_title('Seven models, one split',fontsize=9.5,color=NAVY); ax.grid(axis='x',color=GRID,lw=0.5); ax.set_axisbelow(True)
th=mods[2][1].theta_; ax=fig.add_subplot(gs_[1]); mos=np.full((2*8+4,5*9-1),np.nan)
for k in range(10): r_,c_=divmod(k,5); mos[r_*12:r_*12+8,c_*9:c_*9+8]=th[k].reshape(8,8)
ax.imshow(mos,cmap='Greens',vmin=0,vmax=16); ax.axis('off')
for k in range(10): r_,c_=divmod(k,5); ax.text(c_*9+3.5,r_*12-0.6,str(k),ha='center',va='bottom',fontsize=9,color=NAVY)
ax.set_title('theta_: mean image of each class',fontsize=9.5,color=NAVY,pad=14)
save(fig,'f19111.png'); NUM['best_vs']=float(gs.best_params_['var_smoothing'])
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM)[:3000])

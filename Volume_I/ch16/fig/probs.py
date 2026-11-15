# Problem figures of Chapter 16. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}
sig=lambda z:1/(1+np.exp(-z))
hours=np.arange(1.,9.); passed=np.array([0,0,0,1,0,1,1,1.]); XA=np.c_[np.ones(8),hours]
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def logloss(X,y,w): z=X@w; return float(np.sum(np.logaddexp(0,z)-y*z))
CHK={}
def sv(fig,name):
    import matplotlib.text as mtext
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image; w,h=Image.open(name).size; pw=min(1.55*135,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)

# 16A.1 odds and probabilities
b0,w0=-2.0,0.5; xs=np.array([2.,4.,8.]); ps=sig(b0+w0*xs); x90=(np.log(9)-b0)/w0
A['16A.1']={'p':[round(float(v),4) for v in ps],'odds_x8':round(float(np.exp(b0+w0*8)),3),'x_p90':round(float(x90),3),'odds_ratio':round(float(np.exp(w0)),4)}
fig,ax=small(); xx=np.linspace(0,12,200); ax.plot(xx,sig(b0+w0*xx),color=GREEN,lw=2)
for x_,p_ in zip(xs,ps): ax.plot([x_,x_],[0,p_],color=GRID,lw=0.8); ax.plot(x_,p_,'o',color=NAVY,ms=4.5,zorder=5)
ax.text(0.3,0.9,'$z=-2+0.5x$',fontsize=9,color=GREEN,va='center'); ax.set_xlabel('$x$'); ax.set_ylabel('$p$'); ax.set_xlim(0,12); ax.set_ylim(0,1.02); ax.set_yticks([0,0.5,1]); ax.set_xticks([0,2,4,6,8,10,12]); sv(fig,'p_16A1.png')
# 16A.2 log loss of four predictions
yy=np.array([1,0,1,0]); pp=np.array([0.9,0.2,0.6,0.5]); li=-(yy*np.log(pp)+(1-yy)*np.log(1-pp))
A['16A.2']={'loss_each':[round(float(v),4) for v in li],'J':round(float(li.sum()),4),'J_half':round(float(4*np.log(2)),4)}
fig,ax=small(); ax.bar(range(1,5),li,color=[OR if v else BLUE for v in yy],width=0.6)
for k,(v,p_,y_) in enumerate(zip(li,pp,yy)): ax.text(k+1,v+0.03,f'{v:.2f}',ha='center',fontsize=9,color=NAVY); ax.text(k+1,-0.04,f'$y$={y_}\n$p$={p_:g}',ha='center',fontsize=9,color=GREY,va='top')
ax.set_ylabel('loss $\\ell_i$'); ax.set_xticks([]); ax.set_ylim(0,0.85); ax.set_xlim(0.4,4.6); fig.subplots_adjust(bottom=0.3); sv(fig,'p_16A2.png')
# 16A.3 one gradient step against one Newton step
g=XA.T@(sig(XA@np.zeros(2))-passed); w02=-0.02*g; w05=-0.05*g; wn=np.array([-3.0,2/3]); Hs=XA.T@(XA*0.25); lam=np.linalg.eigvalsh(Hs)
A['16A.3']={'g':[float(v)+0.0 for v in g],'w_eta02':[round(float(v)+0.0,4) for v in w02],'J0':round(logloss(XA,passed,np.zeros(2)),4),'J_eta02':round(logloss(XA,passed,w02),4),'J_eta05':round(logloss(XA,passed,w05),4),'J_newton':round(logloss(XA,passed,wn),4),'lam_max':round(float(lam[-1]),3),'eta_max':round(float(2/lam[-1]),4)}
bg,wg_=np.meshgrid(np.linspace(-6.5,1,160),np.linspace(-0.2,1.8,160)); Jg=np.array([[logloss(XA,passed,np.array([b_,w_])) for b_ in bg[0]] for w_ in wg_[:,0]])
fig,ax=small(2.4,1.8); ax.contourf(bg,wg_,Jg,levels=[2.5,3,4,6,9,14,24,40],cmap='Blues_r',alpha=0.9); arr(ax,(0,0),tuple(w02),c='white',lw=1.4,ms=7); arr(ax,(0,0),tuple(wn),c=GREEN,lw=1.6,ms=8); ax.plot(0,0,'o',color=NAVY,ms=4)
ax.text(-3.2,0.85,'Newton',color='white',fontsize=9,weight='bold',ha='center'); ax.annotate('gradient\nstep',xy=(0,0.14),xytext=(-1.2,1.15),color='white',fontsize=9,ha='center',va='center',arrowprops=dict(arrowstyle='-',color='white',lw=0.7)); ax.set_xlabel('$b$'); ax.set_ylabel('$w$'); sv(fig,'p_16A3.png')
# 16A.4 softmax of one sample
f=np.array([0.,1.,3.]); P=softmax(f); Y=np.array([0.,1.,0.]); A['16A.4']={'P':[round(float(v),4) for v in P],'loss':round(float(-np.log(P[1])),4),'grad':[round(float(v),4) for v in P-Y]}
fig,ax=small(); ax.bar([1,2,3],P,color=[GREY,OR,GREY],width=0.6,edgecolor=[GREY,OR,GREY],alpha=0.9)
for k,v in enumerate(P): ax.text(k+1,v+0.02,f'{v:.3f}',ha='center',fontsize=9,color=NAVY)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['$f_1$=0','$f_2$=1\n(true)','$f_3$=3']); ax.set_ylabel('$P_k$'); ax.set_ylim(0,1.0); fig.subplots_adjust(bottom=0.25); sv(fig,'p_16A4.png')
# 16B.1 odds ratio and a probability threshold with sklearn
m=LogisticRegression(C=np.inf,tol=1e-10,max_iter=1000).fit(hours[:,None],passed); bw,ww=float(m.intercept_[0]),float(m.coef_[0,0]); x80=(np.log(4)-bw)/ww
A['16B.1']={'w':round(ww,4),'b':round(bw,4),'odds_ratio':round(float(np.exp(ww)),3),'x_p80':round(float(x80),3),'p_at_x80':round(float(m.predict_proba([[x80]])[0,1]),4)}
fig,ax=small(); xx=np.linspace(0,9,200); ax.plot(xx,sig(bw+ww*xx),color=GREEN,lw=2); ax.axhline(0.8,color=PUR,lw=1,ls='--'); ax.plot(x80,0.8,'o',color=PUR,ms=5,zorder=5); ax.text(0.3,0.86,'$p$ = 0.8',color=PUR,fontsize=9,va='bottom')
ax.text(x80+0.2,0.35,'$x$ = ?',color=PUR,fontsize=9); ax.plot([x80,x80],[0,0.8],color=PUR,lw=0.8,ls=':'); ax.set_xlabel('hours $x$'); ax.set_ylabel('$P$(pass)'); ax.set_xlim(0,9); ax.set_ylim(0,1.02); ax.set_yticks([0,0.5,1]); sv(fig,'p_16B1.png')
# 16B.2 class_weight on imbalanced data
Xi,yi=make_classification(n_samples=2000,n_features=6,n_informative=3,n_redundant=0,weights=[0.9,0.1],class_sep=1.0,flip_y=0.02,random_state=0)
Xtr,Xte,ytr,yte=train_test_split(Xi,yi,test_size=0.5,random_state=0,stratify=yi); res={}
for name,cw in (('none',None),('balanced','balanced')):
    mm=LogisticRegression(class_weight=cw).fit(Xtr,ytr); pr=mm.predict(Xte); res[name]={'recall':round(float(recall_score(yte,pr)),4),'precision':round(float(precision_score(yte,pr)),4),'acc':round(float(mm.score(Xte,yte)),4)}
cwv=len(ytr)/(2*np.bincount(ytr)); A['16B.2']={'minority_share':round(float(yi.mean()),4),'balanced_weights':[round(float(v),3) for v in cwv],**res}
fig,ax=small(2.4,1.8); wd=0.34
for k,(nm,hat) in enumerate((('none',''),('balanced','//'))):
    vals=[res[nm]['recall'],res[nm]['precision']]; ax.bar(np.arange(2)+(k-0.5)*wd,vals,width=wd,color=[BLUE,OR][k],hatch=hat,edgecolor='white',label='None' if nm=='none' else "'balanced'")
    for i_,v in enumerate(vals): ax.text(i_+(k-0.5)*wd,v+0.02,f'{v:.2f}',ha='center',fontsize=8.5,color=NAVY)
ax.set_xticks([0,1]); ax.set_xticklabels(['recall','precision']); ax.set_ylim(0,1.55); ax.set_yticks([0,0.5,1]); ax.legend(loc='upper center',title='class_weight',fontsize=9,title_fontsize=9,ncol=2,borderaxespad=0.1,columnspacing=1.0,handlelength=1.4); sv(fig,'p_16B2.png')
# 16B.3 elastic net: nonzero weights
bc=load_breast_cancer(); Z=StandardScaler().fit_transform(bc.data); rhos=[0,0.25,0.5,0.75,1.0]; nz=[]
for r in rhos:
    mm=LogisticRegression(C=0.1,l1_ratio=r,solver='saga',max_iter=20000,tol=1e-6,random_state=0).fit(Z,bc.target); nz.append(int((mm.coef_!=0).sum()))
A['16B.3']={'l1_ratio':rhos,'nonzero':nz}
fig,ax=small(2.4,1.8); ax.bar(range(5),nz,color=PUR,width=0.6)
for k,v in enumerate(nz): ax.text(k,v+0.8,str(v),ha='center',fontsize=9,color=NAVY)
ax.set_xticks(range(5)); ax.set_xticklabels(['0','0.25','0.5','0.75','1']); ax.set_xlabel('l1_ratio'); ax.set_ylabel('nonzero weights'); ax.set_ylim(0,36); sv(fig,'p_16B3.png')
# 16C.1 least squares line against logistic curve
cl=np.polyfit(hours,passed,1); fig,ax=small(); xx=np.linspace(-1,10,100); ax.plot(xx,np.polyval(cl,xx),color=GREY,lw=1.4,ls='--'); ax.plot(xx,sig(bw+ww*xx),color=GREEN,lw=1.8)
ax.scatter(hours[passed==0],passed[passed==0],s=22,color=BLUE,edgecolor='white',zorder=5); ax.scatter(hours[passed==1],passed[passed==1],s=22,color=OR,marker='s',edgecolor='white',zorder=5)
ax.set_xlabel('hours $x$'); ax.set_ylabel('$y$'); ax.set_ylim(-0.4,1.5); ax.set_yticks([0,1]); ax.set_xlim(-1,10); sv(fig,'p_16C1.png')
A['16C.1']={'ols_at10':round(float(np.polyval(cl,10)),3)}
# 16C.2 separable data
ysep=np.array([0,0,0,1,1,1,1,1.]); sg=np.linspace(0,12,200); Ju=np.array([np.logaddexp(0,-(2*ysep-1)*s_*(hours-3.5)).sum() for s_ in sg])
fig,ax=small(); ax.plot(sg,Ju,color=RED,lw=1.8); ax.set_xlabel('weight $w$'); ax.set_ylabel('$J$'); ax.set_ylim(0,6); ax.set_xlim(0,12); ax.text(4.5,3.2,'$J\\to 0$ as $w\\to\\infty$',fontsize=9,color=RED); sv(fig,'p_16C2.png')
# 16C.3 hours or minutes
res={}; fig,ax=small(2.4,1.8); xx=np.linspace(0,9,200)
for nm,fac,c,ls in (('hours',1,GREEN,'-'),('minutes',60,OR,'--')):
    mm=LogisticRegression().fit(fac*hours[:,None],passed); res[nm]={'w':round(float(mm.coef_[0,0]),5),'b':round(float(mm.intercept_[0]),4),'boundary':round(float(-mm.intercept_[0]/mm.coef_[0,0]/fac),4),'p6':round(float(mm.predict_proba([[fac*6.0]])[0,1]),4)}
    ax.plot(xx,mm.predict_proba(fac*xx[:,None])[:,1],color=c,ls=ls,lw=1.8,label=nm)
ax.set_xlabel('hours $x$'); ax.set_ylabel('$P$(pass)'); ax.legend(loc='lower right',fontsize=9,title='unit of $x$',title_fontsize=9); sv(fig,'p_16C3.png'); A['16C.3']=res
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1); print(json.dumps(A)); print(CHK)

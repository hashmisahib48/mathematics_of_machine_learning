# Problem figures of Chapter 26. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); sys.path.insert(0,HERE+'/../code'); os.chdir(HERE+'/..')
warnings.simplefilter('error',FutureWarning); warnings.simplefilter('error',DeprecationWarning)
from figstyle import *
import matplotlib.text as mtext
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
plt.rcParams.update({'xtick.labelsize':10.5,'ytick.labelsize':10.5,'axes.labelsize':11,'legend.fontsize':10.5})
A={}; CHK={}
def small(w=1.85,h=1.35): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image
    with Image.open(name) as im: w,h=im.size
    pw=min(1.55*92,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def r4(v): return round(float(v),4)

# 26A.1 one round with a stump that errs on two of five samples
w0=np.full(5,0.2); wrong=np.array([0,1,0,0,1],bool); eps=w0[wrong].sum(); al=np.log((1-eps)/eps)
wn=w0*np.exp(al*wrong); Z=wn.sum(); wn/=Z
A['26A.1']={'eps':r4(eps),'alpha':r4(al),'Z':r4(Z),'w_mistake':r4(wn[wrong][0]),'w_other':r4(wn[~wrong][0]),'err_after':r4(wn[wrong].sum())}
fig,ax=small(); xs=np.arange(5); ax.bar(xs-0.2,w0,0.38,color=GREY,label='before'); ax.bar(xs+0.2,wn,0.38,color=[RED if v else BLUE for v in wrong],label='after')
ax.set_xticks(xs); ax.set_xticklabels([str(k+1) for k in xs]); ax.set_xlabel('sample'); ax.set_ylabel('weight'); ax.set_ylim(0,0.34); grid(ax); ax.legend(loc='upper left',fontsize=10.5,borderaxespad=0.2,handlelength=1.0,ncol=2,columnspacing=0.8); sv(fig,'p_26A1.png')

# 26A.2 three classes, error 1/2, two learning rates
rows={}
for eta in (1.0,0.5):
    a=eta*(np.log(0.5/0.5)+np.log(2)); w=np.full(6,1/6); wr=np.array([1,1,1,0,0,0],bool); w2=w*np.exp(a*wr); Z=w2.sum(); w2/=Z
    rows[str(eta)]={'alpha':r4(a),'w_mistake':r4(w2[0]),'w_other':r4(w2[3]),'share_mistakes':r4(w2[wr].sum())}
A['26A.2']=rows
fig,ax=small(); ax.bar([0,1],[rows['1.0']['share_mistakes'],rows['0.5']['share_mistakes']],0.5,color=[GREEN,PUR]); ax.axhline(0.5,color=GREY,lw=0.9,ls=':')
for k,k_ in enumerate(['1.0','0.5']): ax.text(k,rows[k_]['share_mistakes']+0.015,f"{rows[k_]['share_mistakes']:.3f}",ha='center',fontsize=11,color=NAVY)
ax.set_xticks([0,1]); ax.set_xticklabels(['$\\eta=1$','$\\eta=0.5$']); ax.set_ylabel('mistake weight'); ax.tick_params(labelsize=11); ax.set_ylim(0,0.8); grid(ax); sv(fig,'p_26A2.png')

# 26A.3 a vote of three stumps
al3=np.array([1.0,0.6,0.5]); h3=np.array([1,-1,-1]); Fv=(al3*h3).sum(); f=2*Fv/al3.sum(); P=1/(1+np.exp(-f))
A['26A.3']={'F':r4(Fv),'f':r4(f),'P1':r4(P)}
fig,ax=small(); ax.bar([1,2,3],al3*h3,color=[OR if v>0 else BLUE for v in h3],width=0.55); ax.axhline(0,color=GREY,lw=0.8)
ax.set_xticks([1,2,3]); ax.set_xticklabels(['$h_1$','$h_2$','$h_3$']); ax.set_ylabel('$\\alpha_m h_m(\\mathbf{x})$'); ax.set_ylim(-0.9,1.3); grid(ax); sv(fig,'p_26A3.png')

# 26A.4 one round of AdaBoost.R2
yr=np.array([1,2,4,8.]); pr=np.array([1,1,5,5.]); e=np.abs(yr-pr); D=e.max(); L=e/D; w=np.full(4,0.25); eps=(w*L).sum(); beta=eps/(1-eps); al=np.log(1/beta)
wn=w*beta**(1-L); Z=wn.sum(); wn/=Z
L2=L**2; eps2=(w*L2).sum(); L3=1-np.exp(-L); eps3=(w*L3).sum()
A['26A.4']={'L':[r4(v) for v in L],'eps':r4(eps),'beta':r4(beta),'alpha':r4(al),'w_new':[r4(v) for v in wn],'eps_square':r4(eps2),'eps_exponential':r4(eps3)}
fig,ax=small(); xr=np.arange(1,5); ax.scatter(xr,yr,s=34,color=BLUE,edgecolor='white',lw=0.7,zorder=5); ax.plot([0.5,2.5],[1,1],color=GREEN,lw=1.6); ax.plot([2.5,4.5],[5,5],color=GREEN,lw=1.6); ax.axvline(2.5,color=GREEN,lw=0.6,ls=':')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_xticks(xr); ax.set_ylim(0,9.5); ax.set_xlim(0.5,4.5); grid(ax); sv(fig,'p_26A4.png')

cd=load_breast_cancer(); Xt,Xe,yt,ye=train_test_split(cd.data,cd.target,test_size=0.25,random_state=0,stratify=cd.target)
# 26B.1 the previous stump has weighted error exactly 1/2 under the new weights
m=ABC(n_estimators=10,random_state=0).fit(Xt,yt); n=len(yt); w=np.full(n,1/n); errs=[]
for h,a in zip(m.estimators_,m.estimator_weights_):
    wr=h.predict(Xt)!=yt; w=w*np.exp(a*wr); w/=w.sum(); errs.append(float(w[wr].sum()))
A['26B.1']={'errors_after_update':[r4(v) for v in errs],'max_dev':float(np.max(np.abs(np.array(errs)-0.5)))}
fig,ax=small(); ax.plot(range(1,11),errs,'o',color=RED,ms=4.5); ax.axhline(0.5,color=GREY,lw=0.9,ls=':'); ax.set_ylim(0.3,0.7); ax.set_xlabel('round $m$'); ax.set_ylabel('error under $w^{(m+1)}$'); ax.set_xticks([1,5,10]); grid(ax); sv(fig,'p_26B1.png')

# 26B.2 perfect fit stops the loop
rows={}
for d in (1,2,3,5,None):
    m=ABC(DTC(max_depth=d),random_state=0).fit(Xt,yt); rows[str(d)]={'rounds':len(m.estimators_),'train':r4(m.score(Xt,yt)),'test':r4(m.score(Xe,ye))}
A['26B.2']=rows
fig,ax=small(); ks=['1','2','3','5','None']; ax.bar(range(5),[rows[k]['rounds'] for k in ks],color=PUR,width=0.6); ax.set_xticks(range(5)); ax.set_xticklabels(ks); ax.set_xlabel('max_depth'); ax.set_ylabel('rounds kept'); ax.set_yscale('log'); ax.set_ylim(0.8,80); ax.set_yticks([1,10]); ax.set_yticklabels(['1','10']); grid(ax); sv(fig,'p_26B2.png')

# 26C.2 where the weight sits after 50 rounds with 10% flipped labels
X,y=make_moons(300+10,noise=0.3,random_state=3); X,y=X[:300],y[:300]; rng=np.random.default_rng(3); flip=rng.random(300)<0.10; yn=np.where(flip,1-y,y)
m=ABC(n_estimators=50,random_state=0).fit(X,yn); w=np.full(300,1/300)
for h,a in zip(m.estimators_,m.estimator_weights_): w=w*np.exp(a*(h.predict(X)!=yn)); w/=w.sum()
o=np.argsort(w)[::-1]; A['26C.2']={'flipped':int(flip.sum()),'share_flipped':r4(w[flip].sum()),'top30_flipped':int(flip[o[:30]].sum()),'max_over_mean':r4(w.max()*300)}
fig,ax=small(); ax.bar(range(60),w[o[:60]]*300,color=[RED if flip[i] else BLUE for i in o[:60]],width=0.85); ax.axhline(1,color=GREY,lw=0.9,ls=':'); ax.set_xlabel('sample, sorted by weight'); ax.set_ylabel('weight $\\times\\, n$'); ax.set_xticks([0,20,40,60]); grid(ax); sv(fig,'p_26C1.png')

# 26C.1 the sign of alpha
e=np.linspace(0.03,0.97,300); fig,ax=small(); ax.plot(e,np.log((1-e)/e),color=GREEN,lw=1.6); ax.axhline(0,color=GREY,lw=0.8); ax.axvline(0.5,color=PUR,lw=0.9,ls='--')
ax.set_xlabel('weighted error $\\varepsilon$'); ax.set_ylabel('$\\alpha$'); ax.set_ylim(-3.6,3.6); grid(ax); sv(fig,'p_26C2.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1)
print({k:v for k,v in CHK.items() if v<7.5}); print(json.dumps(A))

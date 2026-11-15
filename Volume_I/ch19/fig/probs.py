# Problem figures of Chapter 19. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
warnings.simplefilter('ignore')
from figstyle import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
A={}
msgs=["free prize now","win free prize","win free now now","meeting report","lunch meeting now","report lunch meeting"]; y=np.array([1,1,1,0,0,0])
vec=CountVectorizer(); X=vec.fit_transform(msgs); words=list(vec.get_feature_names_out())
def ps(text,m): return float(m.predict_proba(vec.transform([text]))[0,1])
# 9A.1 hand = sklearn
mn=MultinomialNB().fit(X,y); N=np.array([X.toarray()[y==k].sum(0) for k in (0,1)],float); th=(N+1)/(N+1).sum(1,keepdims=True)
xq=vec.transform(["lunch now now"]).toarray()[0]; js=np.log([0.5,0.5])+xq@np.log(th).T; hand=float(np.exp(js[1]-js.max())/np.exp(js-js.max()).sum())
assert abs(hand-ps("lunch now now",mn))<1e-12
A['19A.1']={'joint_ham':round(float(np.exp(js[0])),6),'joint_spam':round(float(np.exp(js[1])),6),'P_spam':round(hand,4)}
def strip(ax,vals,words,fs=9,fc=LB,ec=BLUE):
    for j,(v,w) in enumerate(zip(vals,words)):
        ax.add_patch(Rectangle((j*0.5,0),0.5,0.5,fc=(ORL if v>0 else '#FFFFFF'),ec=BLUE,lw=0.8)); ax.text(j*0.5+0.25,0.25,f'{v:g}',ha='center',va='center',fontsize=10,color=NAVY,weight='bold' if v>0 else 'normal')
        ax.text(j*0.5+0.27,-0.08,w,ha='right',va='top',rotation=50,fontsize=9,color=NAVY,rotation_mode='anchor')
ORL='#FCE3BE'
fig,ax=new(2.6,1.7); strip(ax,xq,words); ax.text(1.75,0.78,'message "lunch now now"',ha='center',fontsize=9,color=NAVY); ax.set_xlim(-0.1,3.7); ax.set_ylim(-1.1,0.95); save(fig,'p_19A1.png')
# 9A.2 gaussian toy query
Xg=np.array([[1,2],[2,4],[3,3],[4,5],[6,9],[8,7]],float); yg=np.array([0,0,0,1,1,1]); g=GaussianNB().fit(Xg,yg); q=np.array([[3.5,4.0]])
pq=g.predict_proba(q)[0]; jl=g.predict_joint_log_proba(q)[0]
A['19A.2']={'log_joint':[round(float(v),4) for v in jl],'P_class0':round(float(pq[0]),4),'P_class1':round(float(pq[1]),4),'dist_sq':[round(float(((q-g.theta_[k])**2).sum()),3) for k in (0,1)]}
fig,ax=plt.subplots(figsize=(2.5,1.8),dpi=240); ax.scatter(Xg[:3,0],Xg[:3,1],s=30,color=BLUE,marker='o',edgecolor='white',lw=0.6,zorder=5); ax.scatter(Xg[3:,0],Xg[3:,1],s=32,color=OR,marker='^',edgecolor='white',lw=0.6,zorder=5)
ax.scatter(q[:,0],q[:,1],s=60,marker='D',color=NAVY,zorder=6); ax.text(3.75,3.45,'query',fontsize=9,color=NAVY,va='top'); ax.set_xlabel('$x_1$',fontsize=9.5); ax.set_ylabel('$x_2$',fontsize=9.5); ax.set_xlim(0,9); ax.set_ylim(1,10); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); ax.tick_params(labelsize=9); save(fig,'p_19A2.png')
# 9A.3 bernoulli
bn=BernoulliNB().fit(X,y); A['19A.3']={'P_spam':round(ps("free meeting now",bn),4),'theta_spam':[round(float(v),3) for v in np.exp(bn.feature_log_prob_[1])],'theta_ham':[round(float(v),3) for v in np.exp(bn.feature_log_prob_[0])],'multinomial_P_spam':round(ps("free meeting now",mn),4)}
xb=(vec.transform(["free meeting now"]).toarray()[0]>0).astype(int)
fig,ax=new(2.6,1.7); strip(ax,xb,words); ax.text(1.75,0.78,'present (1) or absent (0)',ha='center',fontsize=9,color=NAVY); ax.set_xlim(-0.1,3.7); ax.set_ylim(-1.1,0.95); save(fig,'p_19A3.png')
# 9B.1 priors
A['19B.1']={}
for pr in ([0.5,0.5],[0.9,0.1],[0.1,0.9]):
    m=MultinomialNB(class_prior=pr).fit(X,y); A['19B.1'][str(pr)]=round(ps("free meeting now",m),4)
m=MultinomialNB(fit_prior=False).fit(X,y); A['19B.1']['fit_prior=False']=round(ps("free meeting now",m),4)
fig,ax=new(3.0,1.5)
box(ax,0,0.9,1.2,0.5,'3 spam,\n3 ham',fc=LGREY,ec=GREY,fs=9); box(ax,1.5,0.9,1.9,0.5,'CountVectorizer',fc=LB,ec=BLUE,fs=9); box(ax,0.55,0.05,2.3,0.55,'MultinomialNB\n(class_prior = ...)',fc='#E3F4EA',ec=GREEN,fs=9)
arr(ax,(1.2,1.15),(1.5,1.15),c=NAVY); arr(ax,(2.45,0.9),(2.1,0.6),c=NAVY); ax.set_xlim(-0.1,3.5); ax.set_ylim(0,1.5); save(fig,'p_19B1.png')
# 9B.2 alpha sweep
A['19B.2']={}
for a in [0.01,0.1,1,10,100]:
    m=MultinomialNB(alpha=a).fit(X,y); A['19B.2'][str(a)]=round(ps("free meeting now",m),4)
fig,ax=plt.subplots(figsize=(2.6,1.3),dpi=240); ax.set_xscale('log'); ax.set_xlim(0.005,200); ax.set_ylim(0,1); ax.set_yticks([]); 
for s_ in ['left']: ax.spines[s_].set_visible(False)
for a in [0.01,0.1,1,10,100]: ax.plot([a,a],[0,0.35],color=PUR,lw=1.5); ax.text(a,0.5,f'{a:g}',ha='center',fontsize=9,color=PUR)
ax.set_xticks([]); ax.text(0.015,0.85,'smoothing  alpha',fontsize=9.5,color=PUR); save(fig,'p_19B2.png')
# 9C.1 duplicate column ; 9C.2 parameter count
fig,ax=new(2.6,1.7)
for j,(lab,c_) in enumerate([('$x$',BLUE),('copy of $x$',BLUE),('$y$',GREEN)]):
    x0=[0.1,0.9,2.1][j]; vals=[0.4,1.2,0.8,1.6][:4]
    for i in range(4): ax.add_patch(Rectangle((x0,1.3-(i+1)*0.28),0.7 if j==1 else 0.6,0.28,fc=LB if j<2 else '#E3F4EA',ec=c_,lw=0.8))
    ax.text(x0+0.3,1.37,lab,ha='center',fontsize=9,color=c_)
ax.text(1.3,0.02,'the same evidence twice',fontsize=9,color=RED,ha='center'); ax.set_xlim(0,3.0); ax.set_ylim(-0.1,1.7); save(fig,'p_19C1.png')
fig,ax=plt.subplots(figsize=(2.4,1.8),dpi=240); p=np.arange(1,31); ax.semilogy(p,2*(2.0**p-1)+1,color=RED,lw=1.5); ax.semilogy(p,2*p+1,color=GREEN,lw=1.5); ax.axvline(30,color=GREY,lw=0.8,ls=':')
ax.set_xlabel('features $p$',fontsize=9.5); ax.set_ylabel('parameters',fontsize=9.5); ax.tick_params(labelsize=9); ax.grid(color=GRID,lw=0.5); ax.set_xlim(0,31); ax.set_yticks([1e1,1e5,1e9]); ax.text(12,3e5,'full table',fontsize=9,color=RED,rotation=0,ha='right'); ax.text(16,1.5e2,'naive',fontsize=9,color=GREEN); save(fig,'p_19C2.png')
A['19C.2']={'p30_full':2*(2**30-1)+1,'p30_naive':61}
# 19A.4 categorical smoothing and an unseen category
cnt=np.array([1.,2.,1.]); th3=(cnt+1)/(cnt.sum()+3); th4=(np.r_[cnt,0]+1)/(cnt.sum()+4)
A['19A.4']={'theta_m3':[round(float(v),4) for v in th3],'theta_m4':[round(float(v),4) for v in th4]}
from sklearn.naive_bayes import CategoricalNB
Xc_=np.array([[0],[1],[1],[2],[0],[2]]); yc_=np.array([1,1,1,1,0,0]); cm_=CategoricalNB(min_categories=4).fit(Xc_,yc_)
assert np.allclose(np.exp(cm_.feature_log_prob_[0][1]),th4)
fig,ax=plt.subplots(figsize=(2.4,1.7),dpi=240); xx=np.arange(4); ax.bar(xx[:3]-0.18,th3,0.36,color=BLUE,label='$m$ = 3'); ax.bar(xx+0.18,th4,0.36,color=PUR,hatch='//',edgecolor='white',label='$m$ = 4')
ax.set_xticks(xx); ax.set_xticklabels(['0','1','2','3 (unseen)'],fontsize=9); ax.set_ylabel('$\\theta$',fontsize=9.5); ax.tick_params(labelsize=9); ax.set_ylim(0,0.6); ax.legend(fontsize=9,loc='upper right',borderaxespad=0.2); ax.set_xlabel('category',fontsize=9.5); save(fig,'p_19A4.png')
# 19B.3 partial_fit on a stream of digits
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
Xd,yd=load_digits(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(Xd,yd,test_size=0.25,random_state=0,stratify=yd)
mp=MultinomialNB(); accs=[]
for idx in np.array_split(np.arange(len(ytr)),5):
    mp.partial_fit(Xtr[idx],ytr[idx],classes=np.arange(10)); accs.append(round(float(mp.score(Xte,yte)),4))
mf=MultinomialNB().fit(Xtr,ytr)
A['19B.3']={'batch_sizes':[len(b) for b in np.array_split(np.arange(len(ytr)),5)],'acc_after_batch':accs,'equal_to_fit':bool(np.allclose(mp.feature_log_prob_,mf.feature_log_prob_)),'fit_acc':round(float(mf.score(Xte,yte)),4)}
fig,ax=plt.subplots(figsize=(2.4,1.7),dpi=240); ax.plot(range(1,6),accs,'o-',color=GREEN,lw=1.5,ms=4); ax.axhline(A['19B.3']['fit_acc'],color=GREY,ls='--',lw=1); ax.text(1.0,A['19B.3']['fit_acc']+0.004,'fit on all',fontsize=9,color=GREY,va='bottom')
ax.set_xlabel('batches seen',fontsize=9.5); ax.set_ylabel('test accuracy',fontsize=9.5); ax.set_xticks(range(1,6)); ax.tick_params(labelsize=9); ax.set_ylim(0.85,0.93); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); save(fig,'p_19B3.png')
# 19C.3 naive Bayes against logistic regression as the training set grows
rgc=np.random.default_rng(0); sizes=[20,40,80,160,320,640,1347]; nb_m=[]; lr_m=[]
for n_ in sizes:
    a1=[];a2=[]
    for r_ in range(10 if n_<1347 else 1):
        ii=rgc.choice(len(ytr),n_,replace=False) if n_<1347 else np.arange(len(ytr))
        a1.append(GaussianNB(var_smoothing=0.1).fit(Xtr[ii],ytr[ii]).score(Xte,yte))
        a2.append(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(Xtr[ii],ytr[ii]).score(Xte,yte))
    nb_m.append(round(float(np.mean(a1)),3)); lr_m.append(round(float(np.mean(a2)),3))
A['19C.3']=None; A['_19C3_curve']={'sizes':sizes,'gaussian_nb':nb_m,'logistic':lr_m}
fig,ax=plt.subplots(figsize=(2.4,1.8),dpi=240); ax.semilogx(sizes,nb_m,'o-',color=BLUE,lw=1.5,ms=3.5,label='GaussianNB'); ax.semilogx(sizes,lr_m,'s--',color=OR,lw=1.5,ms=3.5,label='logistic')
ax.set_xlabel('training images',fontsize=9.5); ax.set_ylabel('test accuracy',fontsize=9.5); ax.tick_params(labelsize=9); ax.legend(fontsize=9,loc='lower right',borderaxespad=0.2); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True); save(fig,'p_19C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

# Problem figures of Chapter 8. Run: python3 fig/probs.py  (answers -> verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from matplotlib.patches import Rectangle, Circle
from sklearn.datasets import load_diabetes, load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
plt.rcParams.update({'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':9.5,'legend.fontsize':9})
A={}; LGB='#DCE6F5'
def small(w=2.6,h=1.8): return plt.subplots(figsize=(w,h),dpi=240)
# 8A.1 four scores
y=np.array([2.,5,7,10]); yh=np.array([3.,4,7,12]); r=y-yh
A['8A.1']={'SSE':float((r**2).sum()),'MSE':float((r**2).mean()),'RMSE':round(float(np.sqrt((r**2).mean())),4),'MAE':float(np.abs(r).mean()),'SST':float(((y-y.mean())**2).sum()),'R2':round(float(r2_score(y,yh)),4),
  'sk':[float(mean_squared_error(y,yh)),round(float(root_mean_squared_error(y,yh)),4),float(mean_absolute_error(y,yh)),round(float(r2_score(y,yh)),4)]}
fig,ax=small(); ax.axhline(y.mean(),color=OR,lw=1.1,ls='--'); ii=np.arange(1,5)
for a,u,v in zip(ii,y,yh): ax.plot([a,a],[u,v],color=RED,lw=2)
ax.scatter(ii,y,s=34,color=BLUE,zorder=5,edgecolor='white',lw=0.7); ax.scatter(ii,yh,s=34,marker='D',facecolor='white',edgecolor=GREEN,lw=1.4,zorder=6)
ax.text(4.45,y.mean()-0.4,'$\\bar y$ = 6',color=OR,fontsize=9,ha='right',va='top'); ax.set_xlim(0.5,4.5); ax.set_ylim(0,13.5); ax.set_xticks(ii); ax.set_xlabel('sample $i$'); ax.set_ylabel('target'); ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.text(0.65,12.8,'● measured   ◇ predicted',fontsize=9,va='top',color=NAVY); save(fig,'p_8A1.png')
# 8A.2 screening
TP,FP,FN=40,20,10; TN=1000-TP-FP-FN
A['8A.2']={'TP':TP,'FP':FP,'FN':FN,'TN':TN,'accuracy':(TP+TN)/1000,'precision':round(TP/(TP+FP),4),'recall':TP/(TP+FN),'F1':round(2*TP/(2*TP+FP+FN),4),'always_healthy_acc':950/1000}
fig,ax=new(2.6,1.8); ax.add_patch(Circle((0.95,0.9),0.78,fc=PINK,ec=RED,lw=1.2,alpha=0.9)); ax.add_patch(Circle((1.85,0.9),0.78,fc=LB,ec=BLUE,lw=1.2,alpha=0.75))
ax.text(0.5,0.9,'10',fontsize=11,ha='center',va='center',color=NAVY); ax.text(1.4,0.9,'40',fontsize=11,ha='center',va='center',color=NAVY,weight='bold'); ax.text(2.2,0.9,'20',fontsize=11,ha='center',va='center',color=NAVY)
ax.text(0.7,1.82,'diseased: 50',fontsize=9,color=RED,ha='center'); ax.text(2.1,1.82,'flagged: 60',fontsize=9,color=BLUE,ha='center'); ax.text(1.4,-0.13,'1000 patients in all',fontsize=9,color=GREY,ha='center')
ax.set_xlim(0,2.9); ax.set_ylim(-0.3,2.0); save(fig,'p_8A2.png')
# 8A.3 leave-one-out by hand
v=np.array([2.,4,9]); loo=[v[k]-np.delete(v,k).mean() for k in range(3)]; tr=v-v.mean()
A['8A.3']={'loo_errors':[float(e) for e in loo],'loo_mse':float(np.mean(np.square(loo))),'train_mse':round(float(np.mean(tr**2)),4),'ratio':float(np.mean(np.square(loo))/np.mean(tr**2)),'sk_loo':float(-cross_val_score(__import__('sklearn.dummy',fromlist=['x']).DummyRegressor(),v[:,None],v,cv=3,scoring='neg_mean_squared_error').mean())}
fig,ax=small(2.6,1.5); ax.axhline(0,color=GRID,lw=0.8)
ax.scatter(v,np.zeros(3),s=46,color=BLUE,zorder=5,edgecolor='white',lw=0.7); ax.axvline(v.mean(),color=OR,lw=1.2,ls='--'); ax.text(v.mean()+0.15,0.55,'mean 5',color=OR,fontsize=9)
for a in v: ax.text(a,-0.45,f'{a:.0f}',ha='center',fontsize=9.5,color=BLUE)
ax.set_xlim(0,10.5); ax.set_ylim(-0.9,1.0); ax.axis('off'); save(fig,'p_8A3.png')
# 8A.4 counts
a=int(np.ceil(0.2*150)); b=[len(te) for _,te in KFold(5).split(np.zeros(103))]; Xi=np.zeros((150,1)); yi=np.repeat([0,1,2],50)
_,_,_,yte=train_test_split(Xi,yi,test_size=0.2,random_state=0,stratify=yi)
cv120=[(len(tr_),len(te_)) for tr_,te_ in KFold(5).split(np.zeros(150-a))]
A['8A.4']={'test150':a,'train150':150-a,'fold103':b,'strat_test':np.bincount(yte).tolist(),'cv120_train_test':cv120[0],'test442':int(np.ceil(0.25*442)),'train442':442-int(np.ceil(0.25*442))}
fig,ax=small(2.6,1.7); ax.bar(range(1,6),b,color=['#7FA6DE']*5,width=0.65,ec='white'); ax.set_ylim(0,26); ax.set_xticks(range(1,6)); ax.set_xlabel('fold'); ax.set_ylabel('test samples'); ax.set_title('KFold(5), $n$ = 103',fontsize=9.5,color=NAVY)
for k,vv in enumerate(b): ax.text(k+1,vv+0.5,str(vv),ha='center',fontsize=9)
save(fig,'p_8A4.png')
# 8B.1 from-scratch CV
Xd,yd=load_diabetes(return_X_y=True)
def cv_scores(model,X,y,k=5):
    out=[]
    for tr_,te_ in KFold(k).split(X):
        m=model.fit(X[tr_],y[tr_]); out.append(r2_score(y[te_],m.predict(X[te_])))
    return np.array(out)
mine=cv_scores(LinearRegression(),Xd,yd); ref=cross_val_score(LinearRegression(),Xd,yd,cv=5)
A['8B.1']={'mine':[round(float(q),3) for q in mine],'sklearn':[round(float(q),3) for q in ref],'equal':bool(np.allclose(mine,ref)),'mean':round(float(mine.mean()),3)}
fig,ax=new(2.6,1.7); box(ax,0.2,1.45,2.4,0.42,'KFold: split the rows',fc=LGREY,ec=GREY,fs=9.5); box(ax,0.2,0.8,2.4,0.42,'fit on the train rows',fc=LB,ec=BLUE,fs=9.5); box(ax,0.2,0.15,2.4,0.42,'$R^2$ on the test rows',fc='#E3F4EA',ec=GREEN,fs=9.5)
arr(ax,(1.4,1.45),(1.4,1.22),c=NAVY); arr(ax,(1.4,0.8),(1.4,0.57),c=NAVY); ax.text(2.75,0.9,'repeat for\neach of\nthe $k$ folds',fontsize=9,ha='left',va='center',color=GREY); ax.set_xlim(0,3.9); ax.set_ylim(0.05,1.95); save(fig,'p_8B1.png')
# 8B.2 expected accuracy of stratified guessing
Xg,yg=load_digits(return_X_y=True); yg=(yg==9).astype(int); Xg_tr,Xg_te,yg_tr,yg_te=train_test_split(Xg,yg,test_size=0.25,random_state=0,stratify=yg)
p1=yg_tr.mean(); theory=p1**2+(1-p1)**2; accs=np.array([DummyClassifier(strategy='stratified',random_state=s).fit(Xg_tr,yg_tr).score(Xg_te,yg_te) for s in range(200)])
A['8B.2']={'p1':round(float(p1),4),'theory':round(float(theory),4),'mean_acc':round(float(accs.mean()),4),'sd':round(float(accs.std()),4),'min':round(float(accs.min()),3),'max':round(float(accs.max()),3)}
fig,ax=small(); ax.hist(accs,bins=14,color=LGB,ec='white'); ax.axvline(theory,color=GREEN,lw=1.6); ax.text(theory-0.004,ax.get_ylim()[1]*0.97,f'{theory:.3f}',color=GREEN,fontsize=9,ha='right',va='top',weight='bold'); ax.set_xlabel('accuracy, 200 seeds'); ax.set_ylabel('count'); save(fig,'p_8B2.png')
# 8B.3 many splits
t=[]
for s in range(20):
    a_,b_,c_,d_=train_test_split(Xd,yd,test_size=0.25,random_state=s); t.append(LinearRegression().fit(a_,c_).score(b_,d_))
t=np.array(t); A['8B.3']={'min':round(float(t.min()),3),'max':round(float(t.max()),3),'mean':round(float(t.mean()),3),'sd':round(float(t.std(ddof=1)),3),'seed0':round(float(t[0]),3)}
fig,ax=small(); ax.scatter(range(20),t,s=18,color=GREY,marker='s',zorder=4); ax.axhline(t.mean(),color=GREEN,lw=1.2); ax.set_xlabel('random_state'); ax.set_ylabel('test $R^2$'); ax.set_ylim(0.2,0.8); ax.grid(color=GRID,lw=0.5); save(fig,'p_8B3.png')
# 8C.1 R2 negative: SSE vs SST, train and new
fig,ax=small(2.6,1.8); lab=['training','new data']; sse_=[1.10,11.14]; sst_=[30.0,5.0]
xx=np.arange(2); ax.bar(xx-0.18,sse_,width=0.34,color=RED,label='SSE of the line'); ax.bar(xx+0.18,sst_,width=0.34,color=OR,label='SST (mean)')
ax.set_xticks(xx); ax.set_xticklabels(lab); ax.set_ylabel('sum of squares'); ax.legend(loc='upper right',fontsize=9,handlelength=1.0); ax.set_ylim(0,38); save(fig,'p_8C1.png')
# 8C.2 peeking
rng=np.random.default_rng(5); n_te=50; sc=rng.binomial(n_te,0.5,size=20)/n_te; A['8C.2']={'best':float(sc.max()),'mean':round(float(sc.mean()),3),'n_test':n_te,'n_models':20}
fig,ax=small(); ax.scatter(range(1,21),sc,s=20,color=GREY,marker='s',zorder=4); ax.axhline(0.5,color=BLUE,lw=1.2); ax.scatter([int(sc.argmax())+1],[sc.max()],s=70,facecolor='none',edgecolor=RED,lw=1.6,zorder=6)
ax.set_title('every model has true accuracy 0.5',fontsize=9,color=BLUE); ax.set_xlabel('model tried'); ax.set_ylabel('test accuracy'); ax.set_ylim(0.34,0.76); ax.grid(color=GRID,lw=0.5); save(fig,'p_8C2.png')
# 8C.3 baseline F1 depends on which class is called positive
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_validate
Xb,yb=load_breast_cancer(return_X_y=True); Xb_tr,_,yb_tr,_=train_test_split(Xb,yb,test_size=0.25,random_state=0,stratify=yb)
res={}
for lab,yy in (('benign = 1',yb_tr),('malignant = 1',1-yb_tr)):
    r_=cross_validate(DummyClassifier(),Xb_tr,yy,cv=5,scoring=['accuracy','f1'])
    res[lab]=(round(float(r_['test_accuracy'].mean()),3),round(float(r_['test_f1'].mean()),3))
A['8C.3']=res
fig,ax=small(2.6,1.8); xx=np.arange(2); acc_=[res[k][0] for k in res]; f1_=[res[k][1] for k in res]
ax.bar(xx-0.18,acc_,width=0.34,color='#7FA6DE',label='accuracy'); ax.bar(xx+0.18,f1_,width=0.34,color=OR,label='$F_1$',hatch='//',ec='white')
for k_ in range(2):
    ax.text(k_-0.18,acc_[k_]+0.03,f'{acc_[k_]:.3f}',ha='center',fontsize=8.5); ax.text(k_+0.18,f1_[k_]+0.03,f'{f1_[k_]:.3f}',ha='center',fontsize=8.5)
ax.set_xticks(xx); ax.set_xticklabels(list(res)); ax.set_ylim(0,1.15); ax.set_ylabel('CV score'); ax.set_title('majority-class baseline',fontsize=9,color=NAVY)
ax.legend(loc='upper right',fontsize=8.5,handlelength=1.0,ncol=2,borderaxespad=0.2); save(fig,'p_8C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

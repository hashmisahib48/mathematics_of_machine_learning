# Problem figures of Chapter 22. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json,warnings; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel
plt.rcParams.update({'xtick.labelsize':10.6,'ytick.labelsize':10.6,'axes.labelsize':10.6,'legend.fontsize':10.6})
A={}; CHK={}
def small(w=2.4,h=1.75): return plt.subplots(figsize=(w,h),dpi=240)
def sv(fig,name):
    import matplotlib.text as mtext
    fs=min(t.get_fontsize() for t in fig.findobj(mtext.Text) if t.get_text().strip() and t.get_visible()); save(fig,name)
    from PIL import Image; w,h=Image.open(name).size; pw=min(1.55*110,235*w/h); CHK[name]=round(fs*pw/96/(w/240),1)
X8=np.array([[1,5],[2,3],[3,4],[4,1],[7,6],[8,2],[4,7],[6,3]],float); y8=np.array([-1,-1,-1,-1,1,1,1,1])
# 22A.1 distance to a plane
w=np.array([3.,4.]); b=-10.; P=np.array([4.,3.]); f=w@P+b; nw=np.linalg.norm(w)
A['22A.1']={'f_P':float(f),'d_P':round(float(f/nw),4),'f_origin':b,'d_origin':round(float(abs(b)/nw),4),'gap':round(float(2/nw),4),'norm_w':float(nw)}
fig,ax=small(); xx=np.array([0,5.0]); ax.plot(xx,(0-b-w[0]*xx)/w[1],color=GREEN,lw=2); ax.plot(xx,(1-b-w[0]*xx)/w[1],color=GREEN,lw=0.9,ls='--'); ax.plot(xx,(-1-b-w[0]*xx)/w[1],color=GREEN,lw=0.9,ls='--')
foot=P-(f/nw**2)*w; ax.plot([P[0],foot[0]],[P[1],foot[1]],color=RED,lw=1.3); ax.scatter(*P,s=30,color=OR,marker='s',zorder=5,edgecolor='white'); ax.scatter(0,0,s=26,color=BLUE,zorder=5,edgecolor='white')
ax.text(4.1,3.25,'(4, 3)',fontsize=10.6,color=NAVY); ax.text(0.15,0.25,'origin',fontsize=10.6,color=NAVY); ax.text(1.55,2.85,'$f=0$',fontsize=10.6,color=GREEN,rotation=-37)
ax.set_xlim(-0.3,5.2); ax.set_ylim(-0.3,4.6); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); sv(fig,'p_22A1.png')
# 22A.2 two points
x1,x2=np.array([1.,2.]),np.array([3.,2.]); al=2/np.sum((x2-x1)**2); wv=al*(x2-x1); bv=1-wv@x2
A['22A.2']={'alpha':float(al),'w':[float(v) for v in wv],'b':float(bv),'margin':float(2/np.linalg.norm(wv)),'f1':float(wv@x1+bv),'f2':float(wv@x2+bv)}
fig,ax=small(); ax.plot([1,3],[2,2],color=GRID,lw=1,ls=':',zorder=1)
ax.scatter(*x1,s=34,color=BLUE,zorder=5,edgecolor='white'); ax.scatter(*x2,s=34,color=OR,marker='s',zorder=5,edgecolor='white'); ax.text(1.08,2.18,'$y=-1$',fontsize=10.6,color=BLUE); ax.text(3.08,2.18,'$y=+1$',fontsize=10.6,color='#B8730A')
ax.set_xlim(0,4.2); ax.set_ylim(0,3.4); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); sv(fig,'p_22A2.png')
# 22A.3 slack of four seeds
z=np.array([1.8,1.0,0.4,-0.5]); xi=np.maximum(0,1-z); C=2.0
A['22A.3']={'xi':[round(float(v),2) for v in xi],'sum_xi':round(float(xi.sum()),2),'C_sum_xi':round(float(C*xi.sum()),2)}
fig,ax=small(); zz=np.linspace(-1.0,2.2,100); ax.plot(zz,np.maximum(0,1-zz),color=RED,lw=1.8)
for k,(zv,xv) in enumerate(zip(z,xi)): ax.scatter(zv,xv,s=30,color=NAVY,zorder=5); ax.text(zv,xv+0.14,'ABCD'[k],ha='center',fontsize=10.6,color=NAVY)
ax.set_xlabel('$y\\,f(\\mathbf{x})$'); ax.set_ylabel('$\\xi$'); ax.set_ylim(-0.1,1.9); ax.set_xlim(-1,2.2); ax.axvline(1,color=GRID,lw=0.8); sv(fig,'p_22A3.png')
# 22A.4 one SMO update
Kii,Kjj,Kij=5.,4.,2.; Ei,Ej=-0.4,0.3; yi,yj=1,-1; ai,aj=0.1,0.1; C4=1.0
eta=Kii+Kjj-2*Kij; ajn=aj+yj*(Ei-Ej)/eta; L=max(0,aj-ai); H=min(C4,C4+aj-ai); ajc=min(max(ajn,L),H); ain=ai+yi*yj*(aj-ajc)
A['22A.4']={'eta':eta,'aj_unclipped':round(ajn,4),'L':L,'H':H,'aj_new':round(ajc,4),'ai_new':round(ain,4),'sum_ay_before':yi*ai+yj*aj,'sum_ay_after':round(yi*ain+yj*ajc,6)}
fig,ax=small(2.0,1.75); t=np.linspace(0,0.5,100); Wc=yj*(Ei-Ej)*(t-aj)-0.5*eta*(t-aj)**2; ax.plot(t,Wc,color=GREEN,lw=1.8)
ax.scatter([aj,ajc],[0,yj*(Ei-Ej)*(ajc-aj)-0.5*eta*(ajc-aj)**2],s=28,color=[NAVY,OR],zorder=5); ax.text(aj+0.01,-0.12,'old',fontsize=10.6,color=NAVY,ha='left'); ax.text(ajc,0.3,'new',fontsize=10.6,color='#B8730A',ha='center')
ax.set_xlabel('$\\alpha_j$'); ax.set_ylabel('$W-W_0$'); ax.set_ylim(-0.4,0.45); sv(fig,'p_22A4.png')
# 22B.1 gamma raw / scaled
bc=load_breast_cancer(); Xb,yb=bc.data,bc.target; Xtr,Xte,ytr,yte=train_test_split(Xb,yb,test_size=0.25,random_state=0,stratify=yb)
sc=StandardScaler().fit(Xtr); Ztr,Zte=sc.transform(Xtr),sc.transform(Xte); p=Xb.shape[1]
res={}
for nm,(a_,b_,c_,d_) in (('raw',(Xtr,ytr,Xte,yte)),('scaled',(Ztr,ytr,Zte,yte))):
    for gm in ('scale','auto'):
        mm=SVC(gamma=gm).fit(a_,b_); res[f'{nm}_{gm}']=round(float(mm.score(c_,d_)),4)
A['22B.1']={'gamma_raw_scale':float(1/(p*Xtr.var())),'gamma_raw_auto':1/p,'gamma_scaled_scale':float(1/(p*Ztr.var())),'acc':res,'var_raw':float(Xtr.var())}
fig,ax=small(2.3,1.29); lab=['raw\nscale','raw\nauto','scaled\nscale','scaled\nauto']; vals=[res['raw_scale'],res['raw_auto'],res['scaled_scale'],res['scaled_auto']]
ax.bar(range(4),vals,color=[GREY,GREY,GREEN,GREEN],width=0.62)
for k,v in enumerate(vals): ax.text(k,v+0.02,f'{v:.2f}',ha='center',fontsize=10.6,color=NAVY)
ax.set_xticks(range(4)); ax.set_xticklabels(lab,fontsize=9.6); ax.set_ylim(0,1.15); ax.set_yticks([0,0.5,1]); ax.set_ylabel('test accuracy'); fig.subplots_adjust(bottom=0.27); sv(fig,'p_22B1.png')
# 22B.3 precomputed kernel
Xm,ym=make_moons(n_samples=200,noise=0.3,random_state=0); Xm_tr,Xm_te,ym_tr,ym_te=train_test_split(Xm,ym,test_size=0.25,random_state=0,stratify=ym)
g=0.5; a1=SVC(kernel='rbf',gamma=g,C=1.0).fit(Xm_tr,ym_tr); a2=SVC(kernel='precomputed',C=1.0).fit(rbf_kernel(Xm_tr,Xm_tr,gamma=g),ym_tr)
f1=a1.decision_function(Xm_te); f2=a2.decision_function(rbf_kernel(Xm_te,Xm_tr,gamma=g))
A['22B.3']={'max_diff':float(np.abs(f1-f2).max()),'same_support':bool(np.array_equal(a1.support_,a2.support_)),'n_sv':int(len(a1.support_)),'acc_rbf':round(float(a1.score(Xm_te,ym_te)),4),'acc_pre':round(float(a2.score(rbf_kernel(Xm_te,Xm_tr,gamma=g),ym_te)),4)}
fig,ax=small(1.72,1.33); ax.scatter(f1,f2,s=10,color=GREEN,zorder=3); lim=[-2.2,2.2]; ax.plot(lim,lim,color=GRID,lw=1,zorder=1); ax.set_xlabel('rbf: $f$'); ax.set_ylabel('precomputed: $f$'); ax.set_xlim(lim); ax.set_ylim(lim); sv(fig,'p_22B2.png')
# 22B.2 SV count vs C
Cs=[0.001,0.01,0.1,1,10,100]; nsv=[]; marg=[]; nerr=[]
for C in Cs:
    mm=SVC(kernel='linear',C=C).fit(Ztr,ytr); nsv.append(int(len(mm.support_))); marg.append(round(float(2/np.linalg.norm(mm.coef_)),3))
A['22B.2']={'C':Cs,'n_sv':nsv,'margin':marg,'n_train':int(len(ytr))}
fig,ax=small(1.46,1.12); ax.semilogx(Cs,nsv,'-o',color=PUR,ms=4,lw=1.5); ax.set_xlabel('$C$'); ax.set_ylabel('support vectors'); ax.set_ylim(0,430); ax.grid(color=GRID,lw=0.5); sv(fig,'p_22B3.png')
# 22C.1 which moves change the street
m8=SVC(kernel='linear',C=1e6,tol=1e-10).fit(X8,y8); w8=m8.coef_[0]; b8=float(m8.intercept_[0])
moves=[(5,np.array([9.,5.])),(3,np.array([5.,2.]))]; A['22C.1']={}
for idx,newp in moves:
    X2=X8.copy(); X2[idx]=newp; m2=SVC(kernel='linear',C=1e6,tol=1e-10).fit(X2,y8)
    A['22C.1'][f'seed{idx+1}']={'f_old':round(float(w8@X8[idx]+b8),3),'f_new_old_line':round(float(w8@newp+b8),3),'new_w':[round(float(v),4) for v in m2.coef_[0]],'new_b':round(float(m2.intercept_[0]),4),'new_sv':[int(v)+1 for v in m2.support_],'margin':round(float(2/np.linalg.norm(m2.coef_)),4)}
fig,ax=small(2.4,1.9)
xx=np.array([0,9.5]); 
for c,ls,lw in ((0,'-',1.5),(1,'--',0.8),(-1,'--',0.8)): ax.plot(xx,(c-b8-w8[0]*xx)/w8[1],color=GREEN,ls=ls,lw=lw)
for cls,col,mk in ((-1,BLUE,'o'),(1,OR,'s')): mm_=y8==cls; ax.scatter(X8[mm_,0],X8[mm_,1],s=16,color=col,marker=mk,edgecolor='white',lw=0.5,zorder=5)
for idx,newp in moves: arr(ax,tuple(X8[idx]),tuple(newp),c=RED,lw=1.2,ms=7); ax.text(*(X8[idx]+np.array([0.15,0.2])),str(idx+1),fontsize=10.6,color=RED)
ax.set_xlim(0,9.6); ax.set_ylim(0,8.2); ax.set_aspect('equal'); ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); sv(fig,'p_22C1.png')
# 22C.2 islands: gamma 100
Xi,yi_=make_moons(n_samples=120,noise=0.3,random_state=1); Xi_tr,Xi_te,yi_tr,yi_te=train_test_split(Xi,yi_,test_size=0.5,random_state=0,stratify=yi_)
mi=SVC(C=100,gamma=100).fit(Xi_tr,yi_tr); A['22C.2']={'train':round(float(mi.score(Xi_tr,yi_tr)),3),'test':round(float(mi.score(Xi_te,yi_te)),3),'n_sv':int(len(mi.support_)),'n':int(len(yi_tr))}
fig,ax=small(2.4,1.9); gx,gy=np.meshgrid(np.linspace(-1.6,2.6,260),np.linspace(-1.2,1.7,260)); F=mi.decision_function(np.c_[gx.ravel(),gy.ravel()]).reshape(gx.shape)
ax.contourf(gx,gy,F,levels=[-1e3,0,1e3],colors=['#DCE8F8','#FBE7C9']); ax.contour(gx,gy,F,levels=[0],colors=GREEN,linewidths=1.2)
for cls,col,mk in ((0,BLUE,'o'),(1,OR,'s')): mm_=yi_tr==cls; ax.scatter(Xi_tr[mm_,0],Xi_tr[mm_,1],s=9,color=col,marker=mk,edgecolor='white',lw=0.4,zorder=5)
ax.set_xlim(-1.6,2.6); ax.set_ylim(-1.2,1.7); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('$C=100,\\ \\gamma=100$'); sv(fig,'p_22C2.png')
# 22C.3 decision values are not probabilities
Xs,ys_=make_moons(n_samples=400,noise=0.35,random_state=3); Xs_tr,Xs_te,ys_tr,ys_te=train_test_split(Xs,ys_,test_size=0.5,random_state=0,stratify=ys_)
ms=SVC(C=1,gamma=1).fit(Xs_tr,ys_tr); fs_=ms.decision_function(Xs_te)
A['22C.3']={'f_range':[round(float(fs_.min()),2),round(float(fs_.max()),2)],'frac_abs_gt1':round(float((np.abs(fs_)>1).mean()),3)}
fig,ax=small(3.4,1.4); bins=np.linspace(-2.6,2.6,27); ax.hist(fs_[ys_te==0],bins=bins,color=BLUE,alpha=0.8,label='class 0'); ax.hist(fs_[ys_te==1],bins=bins,color=OR,alpha=0.8,label='class 1')
ax.axvline(0,color=GREEN,lw=1.2); ax.set_xlabel('decision_function  $f$'); ax.set_ylabel('count'); ax.legend(loc='upper center',fontsize=10.6,ncol=2,borderaxespad=0.1,columnspacing=0.8,handlelength=1.0); ax.set_ylim(0,40); sv(fig,'p_22C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); json.dump(CHK,open('verify/probfig_textsize.json','w'),indent=1); print(json.dumps(A)); print(CHK)

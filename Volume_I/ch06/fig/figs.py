# All figures of Chapter 6. Run: python3 fig/figs.py  (PNGs into ch06/, numbers into verify/fig_numbers.json)
import sys,os,json,runpy,re; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from numpy.polynomial import Polynomial
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
NUM={}
import contextlib,io
def rp(f):
    with contextlib.redirect_stdout(io.StringIO()): return runpy.run_path(f)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def tk(ax,s=8): ax.tick_params(labelsize=s,length=2.5)
def x01(ax): ax.set_xticks([0,0.5,1]); ax.set_xticklabels(['0','0.5','1'])
rm=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
xg=np.linspace(0,1,300); tru=lambda t: np.sin(2*np.pi*t)

# data of Listing 6.2
sw=rp('code/sweep.py'); xtr,ytr,xte,yte=sw['x_tr'],sw['y_tr'],sw['x_te'],sw['y_te']

# ---- 6.0-1 opener: training and test error against degree
degs=list(range(0,15)); tr=[];te=[]
for d in degs:
    f=Polynomial.fit(xtr,ytr,d); tr.append(rm(ytr,f(xtr))); te.append(rm(yte,f(xte)))
NUM['sweep']={'degree':degs,'train':[round(v,3) for v in tr],'test':[round(v,3) for v in te]}
fig,ax=plt.subplots(figsize=(4.9,2.0),dpi=240)
ax.axvspan(-0.5,2.5,color='#E3EEFB',lw=0); ax.axvspan(2.5,8.5,color='#E3F4EA',lw=0); ax.axvspan(8.5,14.5,color='#FBE6E3',lw=0)
ax.plot(degs,te,'s--',color=GREY,ms=3.5,lw=1.3,label='test error (400 new samples)'); ax.plot(degs,tr,'o-',color=BLUE,ms=3.5,lw=1.6,label='training error (15 samples)')
ax.axhline(0.25,color=RED,lw=0.9,ls=':',label='noise level 0.25')
ax.set_yscale('log'); ax.set_ylim(0.04,3e4); ax.set_xlim(-0.5,14.5)
for xc,t,c in ((1,'underfitting',BLUE),(5.5,'about right','#1E6B43'),(11.5,'overfitting',RED)): ax.text(xc,1.2e4,t,ha='center',va='center',fontsize=9,color=c,weight='bold')
ax.annotate('',xy=(11,tr[11]),xytext=(11,te[11]),arrowprops=dict(arrowstyle='<->',color=RED,lw=1.0)); ax.text(11.3,3.2,'generalisation\ngap',fontsize=8.5,color=RED,va='center')
ax.set_xlabel('model complexity: polynomial degree',color=PUR); ax.set_ylabel('RMSE (log scale)'); h,l=ax.get_legend_handles_labels(); ax.legend([h[1],h[0],h[2]],[l[1],l[0],l[2]],fontsize=8.5,loc='center left',bbox_to_anchor=(0.0,0.5),handlelength=1.8); grid(ax); tk(ax); ax.set_xticks(range(0,15,2))
save(fig,'f600.png')

# ---- 6.1-1 Example 6.1-1: line and lookup on the delivery data
gt=rp('code/gap_toy.py'); x,y,xn,yn=gt['x'],gt['y'],gt['x_new'],gt['y_new']
fig,ax=plt.subplots(figsize=(3.25,2.1),dpi=240); tt=np.linspace(0.5,6.5,400)
ax.step(tt,gt['lookup'](tt),where='mid',color=OR,lw=1.5,ls='--',label='lookup (copies nearest)'); ax.plot(tt,gt['line'](tt),color=GREEN,lw=1.8,label='fitted line')
ax.scatter(x,y,s=34,color=BLUE,edgecolor='white',lw=0.8,zorder=5,label='training samples'); ax.scatter(xn,yn,s=36,marker='s',facecolor='white',edgecolor=GREY,lw=1.3,zorder=5,label='new deliveries')
ax.set_xlabel('distance $x$ (km)'); ax.set_ylabel('time $y$ (min)'); ax.set_xlim(0.5,6.6); ax.set_ylim(1,15.5); grid(ax); tk(ax); ax.legend(fontsize=8,loc='upper left',handlelength=1.8)
save(fig,'f611.png')

# ---- 6.2-1 three fits
fig,axs=plt.subplots(1,3,figsize=(4.9,2.0),dpi=240,sharey=True); NUM['panels']={}
for ax,d,t in zip(axs,(1,3,9),('degree 1: too stiff','degree 3','degree 9: too flexible')):
    f=Polynomial.fit(xtr,ytr,d); ax.plot(xg,tru(xg),color=GRID,lw=1.8,zorder=1); ax.scatter(xte[::4],yte[::4],s=7,color=GREY,alpha=0.55,lw=0,zorder=2)
    ax.plot(xg,f(xg),color=GREEN,lw=1.6,zorder=3); ax.scatter(xtr,ytr,s=20,color=BLUE,edgecolor='white',lw=0.6,zorder=5)
    a,b_=rm(ytr,f(xtr)),rm(yte,f(xte)); NUM['panels'][d]=[round(a,3),round(b_,3)]
    ax.set_ylim(-2.4,2.2); ax.set_xlim(0,1); ax.set_title(t,fontsize=8.5,color=NAVY,pad=3); ax.set_xlabel('$x$',fontsize=9); grid(ax); tk(ax,7.5); x01(ax)
    ax.text(0.5,1.75,f'train {a:.2f}  test {b_:.2f}',fontsize=8,ha='center',va='center',color=NAVY)
axs[0].set_ylabel('$y$',fontsize=9); fig.tight_layout(w_pad=0.4); save(fig,'f621.png')

# ---- bias-variance simulation (same design as Listing 6.3), degrees 0..12
bv=rp('code/bias_var.py'); n=bv['n']; sig=bv['sigma']; x_tr=bv['x_tr']; x0=bv['x0']; f0=bv['f0']; ns=bv['n_sets']
rng=np.random.default_rng(600); B2=[];V=[];FAN={}
for d in range(0,13):
    P=np.empty((ns,x0.size))
    for s in range(ns): P[s]=Polynomial.fit(x_tr,tru(x_tr)+rng.normal(0,sig,n),d)(x0)
    B2.append(float(np.mean((Polynomial.fit(x_tr,tru(x_tr),d)(x0)-f0)**2))); V.append(float(np.mean(P.var(0))))   # exact bias: the fit is linear in y
    if d in (1,3,9): FAN[d]=P[:30].copy(); FAN[(d,'mean')]=P.mean(0)
NUM['bv']={'degree':list(range(13)),'bias2':[round(v,4) for v in B2],'var':[round(v,4) for v in V]}
fig,axs=plt.subplots(1,3,figsize=(4.9,1.8),dpi=240,sharey=True)
for ax,d,t in zip(axs,(1,3,9),('degree 1: high bias','degree 3','degree 9: high variance')):
    for row in FAN[d]: ax.plot(x0,row,color=GREEN,lw=0.6,alpha=0.35,zorder=2)
    ax.plot(x0,tru(x0),color=GREY,lw=1.8,zorder=3); ax.plot(x0,FAN[(d,'mean')],color=NAVY,lw=1.5,ls='--',zorder=4)
    ax.set_ylim(-2.2,2.2); ax.set_xlim(0,1); ax.set_title(t,fontsize=8.5,color=NAVY,pad=3); ax.set_xlabel('$x$',fontsize=9); grid(ax); tk(ax,7.5); x01(ax)
axs[0].set_ylabel('$y$',fontsize=9); fig.tight_layout(w_pad=0.4); save(fig,'f631.png')

# ---- 6.3-2 bias^2, variance, total against degree
dd=np.arange(13); tot=np.array(B2)+np.array(V)+sig**2
fig,ax=plt.subplots(figsize=(3.5,2.3),dpi=240)
ax.plot(dd,B2,'o--',color=RED,ms=3.5,lw=1.3,label='bias$^2$'); ax.plot(dd,V,'^:',color=BLUE,ms=4,lw=1.5,label='variance'); ax.axhline(sig**2,color=GREY,lw=1.1,ls='-.',label='noise $\\sigma^2$')
ax.plot(dd,tot,'s-',color=NAVY,ms=3.5,lw=1.8,label='expected test error'); ax.set_yscale('log'); ax.set_ylim(1e-5,3); ax.set_xlim(-0.4,12.4)
ib=int(np.argmin(tot)); ax.plot(ib,tot[ib],'o',mfc='none',mec=PUR,ms=11,mew=1.6); ax.text(ib+0.5,0.45,f'lowest: degree {ib}',color=PUR,fontsize=8.5,va='center')
NUM['bv_best']=ib; ax.set_xlabel('polynomial degree',color=PUR); ax.set_ylabel('error (squared units, log scale)'); ax.text(7.2,3e-5,'bias$^2$ < 10$^{-7}$ from degree 7 on',fontsize=8,color=RED,ha='left'); ax.legend(fontsize=8,loc='center right',bbox_to_anchor=(1.0,0.36),handlelength=1.8); grid(ax); tk(ax)
save(fig,'f632.png')

# ---- 6.4-1 learning curves (same calls as Listing 6.4)
lc=rp('code/learning.py'); X=lc['X']; y=lc['y']; sizes=lc['sizes']; LC={}
for d in (1,3,9):
    m=make_pipeline(PolynomialFeatures(d,include_bias=False),LinearRegression())
    nu,a,b_=learning_curve(m,X,y,train_sizes=sizes,cv=5,shuffle=True,random_state=0,scoring='neg_root_mean_squared_error'); LC[d]=(nu,-a.mean(1),-b_.mean(1))
NUM['lc']={str(d):{'n':LC[d][0].tolist(),'train':[round(v,3) for v in LC[d][1]],'val':[round(v,3) for v in LC[d][2]]} for d in LC}
fig,axs=plt.subplots(1,3,figsize=(4.9,2.1),dpi=240,sharey=True)
for ax,d,t in zip(axs,(1,3,9),('degree 1: high bias','degree 3','degree 9: high variance')):
    nu,a,b_=LC[d]; ax.plot(nu,a,'o-',color=BLUE,ms=3,lw=1.5,label='training'); ax.plot(nu,b_,'s--',color=GREY,ms=3,lw=1.4,label='validation')
    ax.axhline(0.25,color=RED,lw=0.9,ls=':'); ax.set_xscale('log'); ax.set_xticks([15,30,60,120,200]); ax.set_xticklabels(['15','30','60','120','200']); ax.set_xlim(12,240); ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_title(t,fontsize=8.5,color=NAVY,pad=3); ax.set_xlabel('training samples $n$',fontsize=8.5); grid(ax); tk(ax,7.5); ax.set_ylim(0.1,0.65)
axs[0].set_ylabel('RMSE',fontsize=9); axs[2].legend(fontsize=8,loc='upper right',handlelength=1.6); axs[0].text(14,0.265,'noise 0.25',fontsize=7.5,color=RED,va='bottom')
fig.tight_layout(w_pad=0.4); save(fig,'f641.png')

# ---- 6.4-2 the gap formula
txt=open('verify/gap_theory.out').read().strip().split('\n')[1:-1]; R=np.array([[float(v) for v in l.split()] for l in txt])
fig,ax=plt.subplots(figsize=(3.9,2.4),dpi=240); nn=np.linspace(6,170,300); p=4; s2=0.25**2
ax.plot(nn,s2*(1+p/nn),color=GREY,lw=1.6,ls='--',label='test, $\\sigma^2(1+p/n)$'); ax.plot(nn,s2*(1-p/nn),color=BLUE,lw=1.6,label='training, $\\sigma^2(1-p/n)$')
ax.scatter(R[:,0],R[:,3],s=22,marker='s',facecolor='white',edgecolor=GREY,lw=1.2,zorder=5); ax.scatter(R[:,0],R[:,1],s=20,color=BLUE,edgecolor='white',lw=0.6,zorder=5)
ax.axhline(s2,color=RED,lw=0.9,ls=':',label='noise $\\sigma^2$')
ax.set_xlabel('training samples $n$   ($p$ = 4 weights)'); ax.set_ylabel('mean squared error'); ax.set_ylim(0.02,0.105); ax.set_xlim(0,172); ax.legend(fontsize=8,loc='upper right',bbox_to_anchor=(1.0,0.97),handlelength=1.8); grid(ax); tk(ax)
save(fig,'f642.png')

# ---- 6.5-1 gallery of ridge penalties
vc=rp('code/val_curve.py'); Xv,yv,model,alphas=vc['X'],vc['y'],vc['model'],vc['alphas']
rg=np.random.default_rng(630); xf=rg.uniform(0,1,400); yf=tru(xf)+rg.normal(0,0.25,400); NUM['gallery']={}
fig,axs=plt.subplots(1,3,figsize=(4.9,2.0),dpi=240,sharey=True)
for ax,a,t in zip(axs,(1e-8,1e-1,1e2),('$\\alpha$ = 10$^{-8}$: free','$\\alpha$ = 0.1','$\\alpha$ = 100: stiff')):
    model.set_params(ridge__alpha=a).fit(Xv,yv); ax.plot(xg,tru(xg),color=GRID,lw=1.8,zorder=1); ax.plot(xg,model.predict(xg[:,None]),color=GREEN,lw=1.6,zorder=3)
    ax.scatter(Xv[:,0],yv,s=20,color=BLUE,edgecolor='white',lw=0.6,zorder=5); a1=rm(yv,model.predict(Xv)); b1=rm(yf,model.predict(xf[:,None])); NUM['gallery'][f'{a:g}']=[round(a1,3),round(b1,3)]
    ax.set_ylim(-2.2,2.2); ax.set_xlim(0,1); ax.set_title(t,fontsize=8.5,color=PUR,pad=3); ax.set_xlabel('$x$',fontsize=9); grid(ax); tk(ax,7.5); x01(ax)
    ax.text(0.5,1.75,f'train {a1:.2f}  test {b1:.2f}',fontsize=8,ha='center',va='center',color=NAVY)
axs[0].set_ylabel('$y$',fontsize=9); fig.tight_layout(w_pad=0.4); save(fig,'f651.png')

# ---- 6.5-2 validation curve
fig,ax=plt.subplots(figsize=(3.55,2.2),dpi=240); T_=-vc['tr'].mean(1); V_=-vc['va'].mean(1); ib=int(np.argmin(V_)); NUM['vc']={'alpha':alphas.tolist(),'train':T_.round(3).tolist(),'val':V_.round(3).tolist(),'best':float(alphas[ib])}
ax.plot(alphas,V_,'s--',color=GREY,ms=3.5,lw=1.4,label='validation (5-fold)'); ax.plot(alphas,T_,'o-',color=BLUE,ms=3.5,lw=1.6,label='training')
ax.axvline(alphas[ib],color=PUR,lw=1.0,ls='--'); ax.text(alphas[ib]*1.5,0.5,f'best $\\alpha$ = {alphas[ib]:.1f}',color=PUR,fontsize=8.5,va='top')
ax.set_xscale('log'); ax.set_ylim(0.08,0.56); ax.set_xlabel('penalty strength $\\alpha$',color=PUR); ax.set_ylabel('RMSE'); ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(0.0,0.5),handlelength=1.8)
ax.text(2e-8,0.075+0.0,'',fontsize=8); ax.text(3e-8,0.17,'overfitting',fontsize=8.5,color=RED); ax.text(1.5,0.12,'underfitting',fontsize=8.5,color=RED); grid(ax); tk(ax)
save(fig,'f652.png')

# ---- 6.6-1 the protocol
fig,ax=new(4.9,1.95); ax.set_aspect('auto')
ax.add_patch(Rectangle((0,1.2),3.7,0.5,fc='white',ec=BLUE,lw=1.0));
for k in range(5):
    ax.add_patch(Rectangle((0.02+k*0.7316,1.22),0.7316-0.04,0.46,fc=LB if k!=3 else '#FDF0E0',ec=BLUE if k!=3 else OR,lw=0.8)); ax.text(0.02+k*0.7316+0.35,1.45,'valid.' if k==3 else 'train',ha='center',va='center',fontsize=8.5,color=NAVY)
ax.add_patch(Rectangle((3.85,1.2),1.05,0.5,fc=LGREY,ec=GREY,lw=1.0,hatch='///')); ax.text(4.375,1.45,'test',ha='center',va='center',fontsize=9,color=NAVY,weight='bold',bbox=dict(fc='white',ec='none',pad=1.5))
ax.text(1.85,1.88,'training part (split into folds)',ha='center',fontsize=8.8,color=BLUE); ax.text(4.375,1.88,'locked away',ha='center',fontsize=8.8,color=GREY)
box(ax,0.0,0.15,1.75,0.62,'fit, compare, tune\n(choose degree, $\\alpha$)',fc=LB,ec=BLUE,fs=8.6); box(ax,2.0,0.15,1.45,0.62,'final model\nfit on training part',fc='#E3F4EA',ec=GREEN,fs=8.6); box(ax,3.7,0.15,1.2,0.62,'score once,\nreport',fc=LGREY,ec=GREY,fs=8.6)
arr(ax,(1.75,0.46),(2.0,0.46),c=NAVY,lw=1.0,ms=8); arr(ax,(3.45,0.46),(3.7,0.46),c=NAVY,lw=1.0,ms=8); arr(ax,(0.9,1.15),(0.9,0.8),c=NAVY,lw=1.0,ms=8)
arr(ax,(4.375,1.15),(4.375,0.8),c=GREY,lw=1.0,ms=8,ls='--')
ax.set_xlim(-0.05,4.95); ax.set_ylim(0.05,2.05); save(fig,'f661.png')

# ---- 6.6-2 winner's curse
wn=rp('code/winner.py'); acc=wn['acc']; best=wn['best']; fresh=wn['fresh']
fig,ax=plt.subplots(figsize=(3.0,2.0),dpi=240); bins=np.arange(0.28,0.8,0.04)
ax.hist(acc,bins=bins,color=GREY,edgecolor='white',alpha=0.85); ax.axvline(best,color=RED,lw=1.6); ax.axvline(fresh,color=GREEN,lw=1.6,ls='--')
ax.text(best+0.012,ax.get_ylim()[1]*0.93,f'best rule\non test\n{best:.2f}',color=RED,fontsize=8.5,ha='left',va='top'); ax.text(fresh+0.03,ax.get_ylim()[1]*0.93,f'same rule on\nfresh data: {fresh:.2f}',color=GREEN,fontsize=8.5,va='top')
ax.set_xlabel('accuracy on the 50 test samples'); ax.set_ylabel('number of rules'); ax.set_xlim(0.28,0.92); ax.set_ylim(0,ax.get_ylim()[1]*1.08); tk(ax); grid(ax)
save(fig,'f662.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print('figs ok')

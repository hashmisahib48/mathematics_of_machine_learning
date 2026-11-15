# All figures of Chapter 4. Run: python3 fig/figs.py  (PNGs into ch04/, numbers into verify/fig_numbers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from scipy import stats
from matplotlib.patches import Ellipse
NUM={}
plt.rcParams.update({'font.size':9.5,'xtick.labelsize':9,'ytick.labelsize':9,'axes.labelsize':10})
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)

# ---- 4.0-1 opener: sample -> fitted density -> area is probability
rng=np.random.default_rng(4); d=rng.normal(5,1.5,120); mu,sg=stats.norm.fit(d)
fig,ax=plt.subplots(figsize=(4.93,1.72),dpi=240)
ax.hist(d,bins=np.arange(0,10.01,0.5),density=True,color=LB,edgecolor=BLUE,lw=0.6)
xx=np.linspace(0,10,400); pdf=stats.norm(mu,sg).pdf(xx); ax.plot(xx,pdf,color=GREEN,lw=2.2)
a,b_=6.0,8.0; m=(xx>=a)&(xx<=b_); ax.fill_between(xx[m],pdf[m],color=GREEN,alpha=0.30,hatch='///',edgecolor=GREEN,lw=0)
NUM['open']={'mu':round(float(mu),3),'sigma':round(float(sg),3),'P_6_8':round(float(stats.norm(mu,sg).cdf(b_)-stats.norm(mu,sg).cdf(a)),4),'n':120}
ax.plot(d,np.full_like(d,-0.012),'|',color=BLUE,ms=7,mew=0.8,clip_on=False)
ax.text(0.15,0.44,'histogram of a sample\n(blue)',color=BLUE,fontsize=12.5,va='top')
ax.text(7.2,0.275,'fitted density\n(green)',color=GREEN,fontsize=12.5,ha='left',va='bottom',weight='bold')
ax.annotate('shaded area =\n$P(6 \\leq X \\leq 8)$',xy=(7.0,0.04),xytext=(8.2,0.09),color=GREEN,fontsize=12.5,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.8))
ax.set_xlim(0,10); ax.set_ylim(-0.03,0.45); ax.set_xlabel('value $x$',fontsize=12); ax.set_ylabel('density $p(x)$',fontsize=12); ax.tick_params(labelsize=12); ax.set_yticks([0,0.1,0.2,0.3,0.4]); save(fig,'f400.png')

# ---- 4.1-1 two dice sample space
fig,ax=plt.subplots(figsize=(3.43,3.08),dpi=240); ax.set_aspect('equal'); ax.axis('off')
cA=cB=cAB=0
for i in range(1,7):
    for j in range(1,7):
        A=(i+j==7); Bv=(i==6 or j==6); fc='white'; hatch=None
        if A and Bv: fc='#D9C7EA'; cAB+=1
        elif A: fc=LB
        elif Bv: fc='#FDE7C2'
        cA+=A; cB+=Bv
        ax.add_patch(Rectangle((i-0.5,j-0.5),1,1,fc=fc,ec=GREY,lw=0.6))
        ax.text(i,j,str(i+j),ha='center',va='center',fontsize=9.5,color=NAVY,weight='bold' if A else 'normal')
for k in range(1,7): ax.text(k,0.15,str(k),ha='center',fontsize=9); ax.text(0.15,k,str(k),ha='center',va='center',fontsize=9)
ax.text(3.5,-0.35,'first die',ha='center',fontsize=9.5); ax.text(-0.45,3.5,'second die',rotation=90,ha='center',va='center',fontsize=9.5)
ax.set_xlim(-0.7,6.7); ax.set_ylim(-0.7,7.0)
ax.add_patch(Rectangle((7.0,5.1),0.4,0.4,fc=LB,ec=GREY,lw=0.6,clip_on=False)); ax.text(7.55,5.3,'A: sum = 7',fontsize=9,va='center')
ax.add_patch(Rectangle((7.0,4.4),0.4,0.4,fc='#FDE7C2',ec=GREY,lw=0.6,clip_on=False)); ax.text(7.55,4.6,'B: a six shows',fontsize=9,va='center')
ax.add_patch(Rectangle((7.0,3.7),0.4,0.4,fc='#D9C7EA',ec=GREY,lw=0.6,clip_on=False)); ax.text(7.55,3.9,'both A and B',fontsize=9,va='center')
NUM['dice']={'A':cA,'B':cB,'AB':cAB,'union':cA+cB-cAB}
save(fig,'f411.png')

# ---- 4.2-1 Bayes: natural frequencies
N,prev,sens,fpr=10000,0.01,0.90,0.05
D=N*prev; H_=N-D; TP=D*sens; FN=D-TP; FP=H_*fpr; TN=H_-FP
NUM['bayes']={'D':D,'H':H_,'TP':TP,'FN':FN,'FP':FP,'TN':TN,'pos':TP+FP,'post':round(TP/(TP+FP),4)}
fig,ax=plt.subplots(figsize=(5.8,2.7),dpi=240); ax.axis('off'); ax.set_xlim(-0.2,12.3); ax.set_ylim(0,5.4)
def nb(x,y,w,h,t,fc,ec,fs=9.5): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=fs,r=0.05)
nb(0,2.2,1.7,0.9,f'{N:,.0f}\npeople','#F6F7FA',GREY)
nb(2.9,3.7,2.1,0.8,f'{D:.0f} sick','#FDE7C2',OR); nb(2.9,0.7,2.1,0.8,f'{H_:,.0f} healthy',LB,BLUE)
nb(6.1,4.5,2.0,0.6,f'{TP:.0f} test +','#FDE7C2',OR); nb(6.1,3.3,2.0,0.6,f'{FN:.0f} test −','#F6F7FA',GREY)
nb(6.1,1.5,2.0,0.6,f'{FP:.0f} test +',LB,BLUE); nb(6.1,0.3,2.0,0.6,f'{TN:,.0f} test −','#F6F7FA',GREY)
for p,q,t,dx,dy in [((1.7,2.8),(2.9,4.1),'1%',-0.4,0.1),((1.7,2.4),(2.9,1.1),'99%',-0.45,-0.1),((5.0,4.2),(6.1,4.75),'90%',-0.1,0.28),((5.0,3.95),(6.1,3.6),'10%',-0.1,-0.35),((5.0,1.3),(6.1,1.75),'5%',-0.1,0.28),((5.0,0.9),(6.1,0.6),'95%',-0.1,-0.35)]:
    arr(ax,p,q,c=GREY,lw=0.9,ms=6); ax.text((p[0]+q[0])/2+dx,(p[1]+q[1])/2+dy,t,fontsize=9.5,color=NAVY,ha='center',va='center')
bx0=9.3; ax.add_patch(Rectangle((bx0,0.7),0.9,4.0,fc='white',ec=GREY,lw=0.8))
hP=4.0*TP/(TP+FP); ax.add_patch(Rectangle((bx0,0.7+4.0-hP),0.9,hP,fc='#FDE7C2',ec=OR,lw=0.8)); ax.add_patch(Rectangle((bx0,0.7),0.9,4.0-hP,fc=LB,ec=BLUE,lw=0.8))
ax.text(bx0+0.45,4.95,f'{TP+FP:.0f} positives',ha='center',fontsize=9.5,color=NAVY,weight='bold')
ax.text(bx0+1.05,0.7+4.0-hP/2,f'{TP:.0f} sick\n({100*TP/(TP+FP):.1f} %)',fontsize=9.5,color=OR,va='center'); ax.text(bx0+1.05,0.7+(4.0-hP)/2,f'{FP:.0f}\nhealthy',fontsize=9.5,color=BLUE,va='center')
arr(ax,(8.1,4.8),(bx0,4.45),c=OR,lw=0.8); arr(ax,(8.1,1.8),(bx0,1.9),c=BLUE,lw=0.8)
save(fig,'f421.png')

# ---- 4.3-1 discrete distributions
fig,axs=plt.subplots(2,2,figsize=(4.93,3.26),dpi=240)
ax=axs[0,0]; ax.bar([0,1],[0.7,0.3],color=[BLUE,OR],width=0.5,edgecolor='white'); ax.set_xticks([0,1]); ax.set_ylim(0,1); ax.set_title('Bernoulli,  $\\theta$ = 0.3',fontsize=10,color=NAVY); ax.set_ylabel('$P(X = k)$')
for k,v in zip([0,1],[0.7,0.3]): ax.text(k,v+0.03,f'{v}',ha='center',fontsize=9)
bn=stats.binom(5,0.3); ks=np.arange(6); ax=axs[0,1]; ax.bar(ks,bn.pmf(ks),color=BLUE,width=0.6,edgecolor='white'); ax.bar([2],[bn.pmf(2)],color=OR,width=0.6,edgecolor='white')
ax.text(2,bn.pmf(2)+0.015,f'{bn.pmf(2):.4f}',ha='center',fontsize=9); ax.set_title('binomial,  $n$ = 5,  $\\theta$ = 0.3',fontsize=10,color=NAVY); ax.set_xticks(ks); ax.set_ylim(0,0.48)
ax=axs[1,0]; cp=[0.5,0.3,0.2]; ax.bar([1,2,3],cp,color=[BLUE,OR,GREEN],width=0.5,edgecolor='white',hatch=None); ax.set_xticks([1,2,3]); ax.set_ylim(0,0.7); ax.set_title('categorical,  $K$ = 3',fontsize=10,color=NAVY); ax.set_ylabel('$P(X = k)$'); ax.set_xlabel('class $k$')
for k,v in zip([1,2,3],cp): ax.text(k,v+0.02,f'{v}',ha='center',fontsize=9)
ax=axs[1,1]; ks=np.arange(0,15)
ax.bar(ks-0.2,stats.poisson(2).pmf(ks),width=0.4,color=BLUE,label='$\\lambda$ = 2',edgecolor='white'); ax.bar(ks+0.2,stats.poisson(6).pmf(ks),width=0.4,color=PUR,label='$\\lambda$ = 6',edgecolor='white',hatch='///')
ax.set_title('Poisson',fontsize=10,color=NAVY); ax.legend(fontsize=9,loc='upper right'); ax.set_xticks([0,2,4,6,8,10,12,14]); ax.set_xlabel('count $k$'); ax.set_ylim(0,0.30)
for ax in axs.ravel(): grid(ax)
fig.tight_layout(h_pad=1.0,w_pad=1.0); save(fig,'f431.png')

# ---- 4.4-1 normal 68-95-99.7
fig,ax=plt.subplots(figsize=(4.75,2.38),dpi=240); xx=np.linspace(-4,4,600); pdf=stats.norm.pdf(xx)
for k,(al,lab) in zip([3,2,1],[(0.18,'99.73%'),(0.32,'95.45%'),(0.55,'68.27%')]):
    m=np.abs(xx)<=k; ax.fill_between(xx[m],pdf[m],color=GREEN,alpha=al,lw=0)
ax.plot(xx,pdf,color=GREEN,lw=2)
for k,y0,lab in [(1,0.455,'68.27%'),(2,0.505,'95.45%'),(3,0.555,'99.73%')]:
    ax.annotate('',xy=(-k,y0),xytext=(k,y0),arrowprops=dict(arrowstyle='<->',color=NAVY,lw=0.9)); ax.text(0,y0+0.01,lab,ha='center',fontsize=9.5,color=NAVY)
    ax.plot([-k,-k],[0,y0],color=GREY,lw=0.6,ls=':'); ax.plot([k,k],[0,y0],color=GREY,lw=0.6,ls=':')
ax.set_xticks(range(-3,4)); ax.set_xticklabels(['$\\mu-3\\sigma$','$\\mu-2\\sigma$','$\\mu-\\sigma$','$\\mu$','$\\mu+\\sigma$','$\\mu+2\\sigma$','$\\mu+3\\sigma$'],fontsize=9.5)
ax.set_ylim(0,0.60); ax.set_xlim(-4,4); ax.set_yticks([0,0.2,0.4]); ax.set_ylabel('density $p(x)$'); save(fig,'f441.png')
z=stats.norm(); NUM['rule']=[round(float(z.cdf(k)-z.cdf(-k)),4) for k in (1,2,3)]

# ---- 4.4-2 uniform, exponential, gamma
fig,axs=plt.subplots(1,3,figsize=(5.1,1.85),dpi=240)
ax=axs[0]; ax.plot([-0.5,0,0,4,4,4.5],[0,0,0.25,0.25,0,0],color=BLUE,lw=2); ax.fill_between([0,4],0.25,color=LB); ax.set_ylim(0,0.5); ax.set_xlim(-0.5,4.5); ax.set_title('uniform on [0, 4]',fontsize=9.5,color=NAVY)
ax.text(2,0.30,'height 1/4',ha='center',fontsize=9,color=NAVY); ax.set_ylabel('density $p(x)$')
xx=np.linspace(0,5,300); ax=axs[1]
for lam,c,ls in [(0.5,'#B79AD1','-'),(1,PUR,'--'),(2,'#4B2368',':')]: ax.plot(xx,stats.expon(scale=1/lam).pdf(xx),color=c,lw=1.8,ls=ls,label=f'$\\lambda$ = {lam}')
ax.set_title('exponential',fontsize=9.5,color=NAVY); ax.legend(fontsize=9); ax.set_ylim(0,2.05)
xx=np.linspace(0,5,300); ax=axs[2]
for k,c,ls in [(1,'#B79AD1','-'),(3,PUR,'--'),(6,'#4B2368',':')]: ax.plot(xx,stats.gamma(a=k,scale=1/2).pdf(xx),color=c,lw=1.8,ls=ls,label=f'$k$ = {k}')
ax.set_title('gamma,  $\\beta$ = 2',fontsize=9.5,color=NAVY); ax.legend(fontsize=9); ax.set_ylim(0,2.05)
for ax in axs: grid(ax); ax.set_xlabel('$x$')
fig.tight_layout(w_pad=0.8); save(fig,'f442.png')

# ---- 4.5-1 mean and variance: one die, two dice
fig,axs=plt.subplots(1,2,figsize=(5.02,2.02),dpi=240)
for ax,vals,pm,col,tt in [(axs[0],np.arange(1,7),np.full(6,1/6),BLUE,'one die'),(axs[1],np.arange(2,13),None,OR,'sum of two dice')]:
    if pm is None: pm=np.array([sum(1 for i in range(1,7) for j in range(1,7) if i+j==v)/36 for v in vals])
    mu_=(vals*pm).sum(); sd_=np.sqrt(((vals-mu_)**2*pm).sum()); ax.axvspan(mu_-sd_,mu_+sd_,color=LGREY,zorder=0)
    ax.bar(vals,pm,color=col,width=0.7,edgecolor='white'); ax.axvline(mu_,color=GREEN,lw=2)
    ax.set_title(tt,fontsize=10,color=NAVY); ax.set_xlabel('value'); ax.set_xticks(vals); ax.set_ylim(0,max(pm)*1.55); grid(ax)
    ax.text(0.03,0.95,f'$\\mu$ = {mu_:.1f}\n$\\sigma$ = {sd_:.3f}',color=GREEN,fontsize=9,weight='bold',va='top',transform=ax.transAxes)
    NUM['die_'+tt.split()[0]]={'mu':round(float(mu_),4),'sd':round(float(sd_),4)}
axs[0].set_ylabel('$P(X = x)$'); fig.tight_layout(w_pad=1); save(fig,'f451.png')

# ---- 4.6-1 correlation gallery
rng=np.random.default_rng(6); fig,axs=plt.subplots(2,3,figsize=(5.02,3.34),dpi=240); NUM['corr']={}
for ax,rho in zip(axs.ravel(),[-0.9,-0.5,0.0,0.5,0.9,'sq']):
    if rho=='sq': xx=rng.uniform(-2,2,150); yy=xx**2+rng.normal(0,0.25,150); tt='$y = x^2$ + noise'
    else:
        S=np.array([[1,rho],[rho,1]]); xy=rng.multivariate_normal([0,0],S,150); xx,yy=xy[:,0],xy[:,1]; tt=f'$\\rho$ = {rho}'.replace('-','−')
    r_=np.corrcoef(xx,yy)[0,1]; NUM['corr'][str(rho)]=round(float(r_),3)
    ax.scatter(xx,yy,s=9,color=BLUE,edgecolor='white',lw=0.3,alpha=0.9); ax.set_title(tt,fontsize=10,color=NAVY); ax.text(0.05,0.93,f'$r$ = {r_:.2f}'.replace('-','−'),transform=ax.transAxes,fontsize=9.5,color=RED,va='top'); grid(ax)
    ax.set_xlim(-3.4,3.4) if rho!='sq' else ax.set_xlim(-2.4,2.4); ax.set_ylim(-3.2,5.0) if rho!='sq' else ax.set_ylim(-1,6.6)
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(h_pad=0.9,w_pad=0.6); save(fig,'f461.png')

# ---- 4.6-2 multivariate normal ellipses
fig,axs=plt.subplots(1,3,figsize=(5.1,1.94),dpi=240); rng=np.random.default_rng(2)
Ss=[(np.array([[1.,0],[0,1]]),'$\\Sigma$ = [[1, 0], [0, 1]]'),(np.array([[4.,0],[0,0.5]]),'$\\Sigma$ = [[4, 0], [0, 0.5]]'),(np.array([[4.,3],[3,5]]),'$\\Sigma$ = [[4, 3], [3, 5]]')]
NUM['mvn']={}
for ax,(S,tt) in zip(axs,Ss):
    xy=rng.multivariate_normal([0,0],S,300); ax.scatter(xy[:,0],xy[:,1],s=5,color=BLUE,alpha=0.55,lw=0)
    v,V=np.linalg.eigh(S)
    for k,ls in [(1,'-'),(2,'--')]:
        ang=np.degrees(np.arctan2(V[1,1],V[0,1])); ax.add_patch(Ellipse((0,0),2*k*np.sqrt(v[1]),2*k*np.sqrt(v[0]),angle=ang,fill=False,ec=GREEN,lw=1.6,ls=ls))
    for j in (0,1): arr(ax,(0,0),tuple(np.sqrt(v[j])*V[:,j]),c=RED,lw=1.4,ms=7)
    ax.set_xlim(-6.3,6.3); ax.set_ylim(-6.3,6.3); ax.set_aspect('equal'); ax.set_title(tt,fontsize=9,color=NAVY); ax.set_xticks([-4,0,4]); ax.set_yticks([-4,0,4]); grid(ax)
    NUM['mvn'][tt]=[round(float(t),4) for t in v]
axs[0].set_ylabel('$x_2$'); [a.set_xlabel('$x_1$') for a in axs]
fig.tight_layout(w_pad=0.5); save(fig,'f462.png')

# ---- 4.7-1 law of large numbers; the n versus n-1 divisor
rng=np.random.default_rng(1); fig,axs=plt.subplots(1,2,figsize=(5.1,2.11),dpi=240)
ax=axs[0]; nn=np.arange(1,3001)
for s,c in zip(range(5),[BLUE,OR,GREEN,PUR,RED]): f=rng.integers(0,2,3000); ax.plot(nn,np.cumsum(f)/nn,color=c,lw=0.9,ls=['-','--','-.',':','-'][s])
ax.fill_between(nn,0.5-1/np.sqrt(nn),0.5+1/np.sqrt(nn),color=LGREY,zorder=0,lw=0); ax.axhline(0.5,color=GREY,lw=0.8)
ax.set_xscale('log'); ax.set_ylim(0,1); ax.set_xlabel('number of tosses $n$'); ax.set_ylabel('fraction of heads'); ax.set_title('five runs of fair-coin tosses',fontsize=9.5,color=NAVY); grid(ax)
ax.text(3000,0.04,'grey band: $\\pm\\, 2 \\times$ standard error',fontsize=9,color=GREY,ha='right')
ax=axs[1]; ns=np.arange(2,31); m0=[];m1=[]
for n in ns: Z=rng.normal(0,1,(20000,n)); m0.append(Z.var(axis=1,ddof=0).mean()); m1.append(Z.var(axis=1,ddof=1).mean())
ax.plot(ns,(ns-1)/ns,color=RED,lw=1.2,ls='--'); ax.plot(ns,m0,'o',color=RED,ms=3,label='divisor $n$'); ax.plot(ns,m1,'s',color=GREEN,ms=3,label='divisor $n-1$'); ax.axhline(1,color=GREY,lw=0.8)
ax.set_ylim(0.4,1.1); ax.set_xlabel('sample size $n$'); ax.set_ylabel('average estimate'); ax.set_title('true variance = 1',fontsize=9.5,color=NAVY); ax.legend(fontsize=9,loc='lower right'); grid(ax)
fig.tight_layout(w_pad=1); save(fig,'f471.png')

# ---- 4.8-1 likelihood of a coin
fig,axs=plt.subplots(1,2,figsize=(5.1,2.11),dpi=240); th=np.linspace(0.001,0.999,999)
def LL(k,n): return k*np.log(th)+(n-k)*np.log(1-th)
ax=axs[0]
for (k,n),c,ls,lab in [((7,10),BLUE,'-','7 of 10'),((70,100),OR,'--','70 of 100')]:
    L_=np.exp(LL(k,n)-LL(k,n).max()); ax.plot(th,L_,color=c,lw=1.8,ls=ls,label=lab)
ax.axvline(0.7,color=GREEN,lw=1.2); ax.plot([0.5],[np.exp(LL(7,10)[np.argmin(abs(th-0.5))]-LL(7,10).max())],'o',color=RED,ms=4)
ax.text(0.46,0.5,'$\\theta$ = 0.5',color=RED,fontsize=9,ha='right'); ax.text(0.72,1.12,'$\\hat\\theta$ = 0.7',color=GREEN,fontsize=9.5,ha='left',weight='bold'); ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(0,1.02)); ax.set_ylim(0,1.55); ax.set_yticks([0,0.25,0.5,0.75,1.0])
ax.set_xlabel('$\\theta$'); ax.set_ylabel('$L(\\theta)$ relative to its maximum'); grid(ax)
ax=axs[1]; l=LL(7,10); ax.plot(th,l,color=BLUE,lw=1.8); i7=np.argmax(l); ax.plot([th[i7]],[l[i7]],'o',color=GREEN,ms=6,zorder=5)
ax.annotate('maximum at $\\hat\\theta$ = 0.7\nslope = 0',xy=(0.7,l[i7]),xytext=(0.22,-17.5),fontsize=9,color=GREEN,arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.8))
ax.set_ylim(-22,0); ax.set_xlabel('$\\theta$'); ax.set_ylabel('$\\ln L(\\theta)$   (7 heads in 10)'); grid(ax)
NUM['coin']={'Lmax':round(float(np.exp(l[i7])),6),'lnLmax':round(float(l[i7]),4),'ratio_half':round(float(0.5**10/(0.7**7*0.3**3)),4)}
fig.tight_layout(w_pad=1); save(fig,'f481.png')

# ---- 4.9-1 posteriors
fig,axs=plt.subplots(1,2,figsize=(5.1,2.11),dpi=240)
ax=axs[0]; th=np.linspace(0,1,500); pr=stats.beta(2,2).pdf(th); po=stats.beta(9,5).pdf(th); lik=th**7*(1-th)**3; lik=lik/np.trapezoid(lik,th)
ax.plot(th,pr,color=GREY,lw=1.5,ls='--',label='prior Beta(2, 2)'); ax.plot(th,lik,color=BLUE,lw=1.5,ls=':',label='likelihood'); ax.plot(th,po,color=GREEN,lw=2.2,label='posterior Beta(9, 5)')
ax.axvline(8/12,color=GREEN,lw=0.8,ls='-'); ax.set_ylim(0,6.2); ax.set_xlabel('$\\theta$'); ax.set_ylabel('density'); ax.legend(fontsize=9,loc='upper left'); grid(ax); ax.text(8/12+0.03,5.2,'MAP 0.667',color=GREEN,fontsize=9,ha='left')
ax=axs[1]; xx=np.linspace(-3.5,6,500)
ax.plot(xx,stats.norm(0,1).pdf(xx),color=GREY,lw=1.5,ls='--',label='prior  N(0, 1)'); ax.plot(xx,stats.norm(3,1).pdf(xx),color=BLUE,lw=1.5,ls=':',label='likelihood  N(3, 1)'); ax.plot(xx,stats.norm(1.5,np.sqrt(0.5)).pdf(xx),color=GREEN,lw=2.2,label='posterior  N(1.5, 0.5)')
ax.set_ylim(0,1.25); ax.set_xlabel('$\\mu$'); ax.legend(fontsize=9,loc='upper left'); grid(ax)
fig.tight_layout(w_pad=1); save(fig,'f491.png')

# ---- 4.10-1 bias and variance of a shrunk mean
fig,axs=plt.subplots(1,2,figsize=(5.1,2.11),dpi=240); mu,sd,n=1.0,2.0,4; se=sd/np.sqrt(n)
ax=axs[0]; xx=np.linspace(-3,5,500)
ax.plot(xx,stats.norm(mu,se).pdf(xx),color=BLUE,lw=2,label='$\\bar{x}$'); ax.plot(xx,stats.norm(0.5*mu,0.5*se).pdf(xx),color=OR,lw=2,ls='--',label='0.5 $\\bar{x}$')
ax.axvline(mu,color=GREY,lw=1,ls=':'); ax.text(mu+0.1,1.25,'true $\\mu$ = 1',fontsize=9,color=GREY); ax.set_xlabel('estimate'); ax.set_ylabel('density'); ax.legend(fontsize=9,loc='upper left'); ax.set_ylim(0,1.5); grid(ax)
ax=axs[1]; c=np.linspace(0,1.2,200); bias2=((c-1)*mu)**2; var=c**2*sd**2/n
ax.plot(c,bias2,color=RED,lw=1.6,ls='--',label='bias$^2$'); ax.plot(c,var,color=BLUE,lw=1.6,ls=':',label='variance'); ax.plot(c,bias2+var,color=NAVY,lw=2.2,label='MSE')
ax.plot([0.5],[0.5],'o',color=GREEN,ms=6,zorder=6); ax.plot([1],[1],'o',color=NAVY,ms=5)
ax.annotate('minimum, $c$ = 0.5',xy=(0.5,0.53),xytext=(0.03,1.55),fontsize=9,color=GREEN,ha='left',arrowprops=dict(arrowstyle='-',color=GREEN,lw=0.8))
ax.annotate('$\\bar{x}$ itself',xy=(1.0,1.05),xytext=(0.72,1.75),fontsize=9,color=NAVY,ha='left',arrowprops=dict(arrowstyle='-',color=NAVY,lw=0.8))
ax.set_xlabel('shrink factor $c$  (estimate $c\\,\\bar{x}$)'); ax.set_ylim(0,2.1); grid(ax)
ax.text(0.04,0.4,'bias$^2$',color=RED,fontsize=9.5); ax.text(0.80,0.30,'variance',color=BLUE,fontsize=9.5); ax.text(0.12,1.17,'MSE',color=NAVY,fontsize=9.5,weight='bold')
fig.tight_layout(w_pad=1); save(fig,'f4101.png')
json.dump(NUM,open('verify/fig_numbers.json','w'),indent=1); print(json.dumps(NUM))

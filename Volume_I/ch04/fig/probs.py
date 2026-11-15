# Problem figures of Chapter 4. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
plt.rcParams.update({'font.size':9.5,'xtick.labelsize':9,'ytick.labelsize':9})
A={}
def small(w=2.46,h=1.3): return plt.subplots(figsize=(w,h),dpi=240)
def grid(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 4A.1 two draws without replacement from 3 red, 2 blue
pr=3/5*2/4; A['4A.1']={'P_both_red':round(pr,4),'P_second_red_given_first_red':0.5,'P_second_red':round(3/5*2/4+2/5*3/4,4)}
fig,ax=new(2.79,1.89); ax.set_xlim(-0.2,7.4); ax.set_ylim(0,4.6)
def nb(x,y,w,h,t,fc,ec): box(ax,x,y,w,h,t,fc=fc,ec=ec,fs=8.5,r=0.05)
nb(0.05,1.9,0.9,0.7,'start','#F6F7FA',GREY)
nb(2.0,3.3,1.3,0.7,'1st red','#FDE7C2',OR); nb(2.0,0.5,1.3,0.7,'1st blue',LB,BLUE)
nb(5.2,3.9,1.5,0.5,'2nd red','#FDE7C2',OR); nb(5.2,2.9,1.5,0.5,'2nd blue',LB,BLUE); nb(5.2,1.1,1.5,0.5,'2nd red','#FDE7C2',OR); nb(5.2,0.1,1.5,0.5,'2nd blue',LB,BLUE)
for p,q,t in [((0.95,2.4),(2.0,3.6),'3/5'),((0.95,2.0),(2.0,1.0),'2/5'),((3.3,3.8),(5.2,4.1),'2/4'),((3.3,3.5),(5.2,3.2),'2/4'),((3.3,1.0),(5.2,1.3),'3/4'),((3.3,0.7),(5.2,0.4),'1/4')]:
    arr(ax,p,q,c=GREY,lw=0.9,ms=6); ax.text((p[0]+q[0])/2-0.15,(p[1]+q[1])/2+0.3*(1 if q[1]>p[1] else -1),t,fontsize=9,color=NAVY,ha='center',va='center')
save(fig,'p_4A1.png')
# 4A.2 Bayes with prevalence 2%, sensitivity 95%, false-positive rate 10%
prev,sens,fpr=0.02,0.95,0.10; pp=sens*prev+fpr*(1-prev); A['4A.2']={'P_pos':round(pp,4),'P_D_given_pos':round(sens*prev/pp,4)}
fig,ax=small(2.46,1.3); N=1000; tp=N*prev*sens; fp=N*(1-prev)*fpr
ax.barh([0],[tp],color=OR,height=0.5,edgecolor='white'); ax.barh([0],[fp],left=[tp],color=BLUE,height=0.5,edgecolor='white')
ax.text(tp/2,0.42,f'{tp:.0f} sick',ha='center',fontsize=9,color=OR); ax.text(tp+fp/2,-0.42,f'{fp:.0f} healthy',ha='center',fontsize=9,color=BLUE)
ax.set_xlim(0,130); ax.set_ylim(-0.8,0.9); ax.set_yticks([]); ax.set_xlabel('positive results among 1000 people'); ax.spines['left'].set_visible(False); save(fig,'p_4A2.png')
A['4A.2'].update({'tp_per_1000':round(tp,1),'fp_per_1000':round(fp,1)})
# 4A.3 covariance and correlation by hand
xa=np.array([1.,2,3,4]); ya=np.array([2.,1,4,3]); C=np.cov(xa,ya); A['4A.3']={'Sxy':float(((xa-xa.mean())*(ya-ya.mean())).sum()),'Sxx':float(((xa-xa.mean())**2).sum()),'Syy':float(((ya-ya.mean())**2).sum()),'cov':round(float(C[0,1]),4),'r':round(float(np.corrcoef(xa,ya)[0,1]),4)}
fig,ax=small(2.46,1.3); ax.scatter(xa,ya,s=34,color=BLUE,edgecolor='white',zorder=5); ax.axvline(2.5,color=GREY,lw=0.8,ls=':'); ax.axhline(2.5,color=GREY,lw=0.8,ls=':'); grid(ax)
for a,b in zip(xa,ya): ax.text(a+(0.12 if a!=2 else -0.12),b-0.3,f'({a:.0f}, {b:.0f})',fontsize=9,color=GREY,ha='left' if a!=2 else 'right')
ax.set_xlim(0.4,4.9); ax.set_ylim(0.4,4.8); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); save(fig,'p_4A3.png')
# 4A.4 normal tail
m,s=170,8; A['4A.4']={'P_gt_180':round(float(stats.norm(m,s).sf(180)),4),'z':1.25,'q90':round(float(stats.norm(m,s).ppf(0.9)),2)}
fig,ax=small(2.46,1.3); xx=np.linspace(145,195,300); ax.plot(xx,stats.norm(m,s).pdf(xx),color=GREEN,lw=1.8); mk=xx>=180; ax.fill_between(xx[mk],stats.norm(m,s).pdf(xx[mk]),color=GREEN,alpha=0.35,hatch='///',edgecolor=GREEN,lw=0)
ax.text(181,0.04,'$P(X > 180)$',fontsize=9,color=GREEN); ax.set_xlabel('height $x$ (cm)'); ax.set_ylabel('density'); ax.set_ylim(0,0.06); ax.set_yticks([0,0.02,0.04]); ax.set_xlim(145,195); save(fig,'p_4A4.png')
# 4B.1 Poisson
po=stats.poisson(3); rng=np.random.default_rng(0); sm=rng.poisson(3,100000)
A['4B.1']={'P2':round(float(po.pmf(2)),4),'P_ge1':round(float(po.sf(0)),4),'sim_P2':round(float((sm==2).mean()),3),'sim_P_ge1':round(float((sm>=1).mean()),3)}
fig,ax=small(2.46,1.3); ks=np.arange(0,11); ax.bar(ks,po.pmf(ks),color=BLUE,width=0.6,edgecolor='white'); ax.bar([2],[po.pmf(2)],color=OR,width=0.6,edgecolor='white'); ax.set_xlabel('calls in one minute $k$'); ax.set_ylabel('$P(N = k)$'); grid(ax); ax.set_xticks(ks[::2]); save(fig,'p_4B1.png')
# 4B.2 normal MLE by optimiser
d=np.array([5.,7,8,10,10])
def nll(p): mu,ls=p; v=np.exp(ls); return 0.5*len(d)*np.log(2*np.pi*v)+((d-mu)**2).sum()/(2*v)
res=minimize(nll,[0.0,0.0],method='Nelder-Mead',options={'xatol':1e-8,'fatol':1e-10,'maxiter':2000}); A['4B.2']={'mu':round(float(res.x[0]),4),'sigma2':round(float(np.exp(res.x[1])),4),'formula_mu':float(d.mean()),'formula_sigma2_n':float(d.var()),'sigma2_n_minus_1':float(d.var(ddof=1))}
fig,ax=small(2.46,1.3); mus=np.linspace(5,11,150); v=np.linspace(1,9,150); M,V=np.meshgrid(mus,v)
LL=-0.5*len(d)*np.log(2*np.pi*V)-((d[None,None,:]-M[...,None])**2).sum(-1)/(2*V); ax.contourf(M,V,LL,levels=14,cmap='Blues'); ax.plot([8],[3.6],'*',color=GREEN,ms=11,mec='white'); ax.set_xlabel('$\\mu$'); ax.set_ylabel('$\\sigma^2$'); ax.text(8.3,3.9,'maximum',fontsize=9,color=GREEN,weight='bold'); save(fig,'p_4B2.png')
# 4B.3 ridge on a constant
y=np.array([2.,3,3,4]); X=np.ones((4,1)); als=[0,1,4,16]; A['4B.3']={str(a):round(float(Ridge(alpha=a,fit_intercept=False).fit(X,y).coef_[0]),4) for a in als}
fig,ax=small(2.46,1.3); aa=np.linspace(0,20,200); ax.plot(aa,12/(4+aa),color=PUR,lw=1.8); ax.plot(als,[12/(4+a) for a in als],'o',color=PUR,ms=4); ax.set_xlabel('$\\alpha$'); ax.set_ylabel('fitted value $\\hat{m}$'); ax.set_ylim(0,3.4); grid(ax); save(fig,'p_4B3.png')
# 4C.1 uncorrelated but dependent
rng=np.random.default_rng(3); xs=rng.uniform(-2,2,120); ys=xs**2+rng.normal(0,0.2,120); A['4C.1']={'r':round(float(np.corrcoef(xs,ys)[0,1]),3)}
fig,ax=small(2.46,1.1); ax.scatter(xs,ys,s=10,color=BLUE,edgecolor='white',lw=0.3); ax.text(0.03,0.95,f'$r$ = {A["4C.1"]["r"]:.2f}'.replace('-','−'),transform=ax.transAxes,color=RED,fontsize=9,va='top'); ax.set_xlabel('$x$'); ax.set_ylabel('$y = x^2$ + noise'); grid(ax); ax.set_ylim(-0.6,5); save(fig,'p_4C1.png')
# 4C.2 posterior probability against prevalence
fig,ax=small(2.46,1.1); pv=np.logspace(-4,0,200); post=0.9*pv/(0.9*pv+0.05*(1-pv)); ax.plot(pv,post,color=GREEN,lw=1.8); ax.set_xscale('log'); ax.plot([0.01],[0.9*0.01/(0.9*0.01+0.05*0.99)],'o',color=OR,ms=5); ax.set_xlabel('prevalence $P(D)$',fontsize=10.5); ax.set_ylabel('$P(D \\mid +)$',fontsize=10.5); ax.tick_params(labelsize=10); grid(ax); ax.set_ylim(0,1.02); ax.text(0.00012,0.5,'orange: 1 % prevalence\n(§4.2)',fontsize=10.5,color=OR); save(fig,'p_4C2.png')
# 4C.3 posterior weight on the data
fig,ax=small(2.46,1.1); nn=np.arange(1,101); ax.plot(nn,nn/(nn+4),color=GREEN,lw=1.8); ax.set_xlabel('number of samples $n$',fontsize=10.5); ax.set_ylabel('weight on\ndata mean',fontsize=10.5); ax.tick_params(labelsize=10); ax.set_ylim(0,1.02); grid(ax); ax.text(22,0.4,'$n\\,/\\,(n + \\sigma^2/\\tau^2)$,  $\\sigma^2/\\tau^2$ = 4',fontsize=10.5,color=GREEN); save(fig,'p_4C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

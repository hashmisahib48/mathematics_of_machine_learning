# Problem figures of Chapter 14. Run: python3 fig/probs.py  (answers into verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor, Ridge
A={}
dP=lambda y,m:2*(y*np.log(np.where(y>0,y,1)/m)-(y-m)); dG=lambda y,m:2*((y-m)/m-np.log(y/m))
def small(w=2.4,h=1.3): return plt.subplots(figsize=(w,h),dpi=240)
def gr(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
# 14A.1 likelihood of counts 2,0,3
yc=np.array([2.,0,3]); mh=yc.mean(); A['14A.1']={'mu_hat':round(float(mh),4),'d':[round(float(v),4) for v in dP(yc,mh)],'D':round(float(dP(yc,mh).sum()),4)}
fig,ax=small(); mm=np.linspace(0.4,4.5,200); ll=yc.sum()*np.log(mm)-3*mm; ax.plot(mm,ll,color=BLUE,lw=1.6); ax.plot(mh,yc.sum()*np.log(mh)-3*mh,'*',color=GREEN,ms=11,mec='white'); ax.axvline(mh,color=GREEN,lw=0.8,ls='--')
ax.text(mh+0.12,-6.6,'$\\hat\\mu$',fontsize=10,color=GREEN); ax.set_xlabel('mean $\\mu$',fontsize=9.5); ax.set_ylabel('$\\ell(\\mu)$ + const',fontsize=9.5); gr(ax); ax.set_ylim(-7,-2); save(fig,'p_14A1.png')
# 14A.2 unit deviances y=3
y0=3.; mu=np.linspace(1.2,6,200); A['14A.2']={'pois_2':round(float(dP(y0,2.)),4),'pois_4.5':round(float(dP(y0,4.5)),4),'gam_2':round(float(dG(y0,2.)),4),'gam_4.5':round(float(dG(y0,4.5)),4)}
fig,ax=small(); ax.plot(mu,dP(y0,mu),color=BLUE,lw=1.6,label='Poisson'); ax.plot(mu,dG(y0,mu),color=OR,lw=1.6,ls='--',label='gamma'); ax.axvline(y0,color=GREEN,lw=0.8)
for m_ in (2,4.5): ax.plot([m_],[dP(y0,m_)],'o',color=BLUE,ms=4); ax.plot([m_],[dG(y0,m_)],'o',color=OR,ms=4)
ax.set_xlabel('prediction $\\mu$',fontsize=9.5); ax.set_ylabel('$d(3,\\mu)$',fontsize=9.5); ax.legend(fontsize=8.5,loc='upper center'); gr(ax); ax.set_ylim(0,1.5); save(fig,'p_14A2.png')
# 14A.3 Newton step, x=0,1,2 y=1,3,5
x3=np.array([0.,1,2]); y3=np.array([1.,3,5]); X3=np.c_[np.ones(3),x3]; b0=np.array([np.log(3),0.]); mu0=np.exp(X3@b0); g=X3.T@(y3-mu0); H=X3.T@(mu0[:,None]*X3); dl=np.linalg.solve(H,g); b1=b0+dl
pm=PoissonRegressor(alpha=0,tol=1e-12).fit(x3[:,None],y3)
A['14A.3']={'g':g.tolist(),'H':H.tolist(),'delta':[round(float(v),4) for v in dl],'b1':round(float(b1[0]),4),'w1':round(float(b1[1]),4),'D0':round(float(dP(y3,mu0).sum()),4),'D1':round(float(dP(y3,np.exp(X3@b1)).sum()),4),'b_final':round(float(pm.intercept_),4),'w_final':round(float(pm.coef_[0]),4),'D_final':round(float(dP(y3,pm.predict(x3[:,None])).sum()),4)}
fig,ax=small(); xx=np.linspace(-0.3,2.3,100); ax.plot(xx,np.full(100,3.),color=GREY,ls=':',lw=1.4); ax.plot(xx,np.exp(b1[0]+b1[1]*xx),color=GREEN,lw=1.7); ax.scatter(x3,y3,s=34,color=BLUE,edgecolor='white',zorder=5); gr(ax)
ax.text(-0.2,3.25,'start $\\mu$ = 3',fontsize=8.5,color=GREY); ax.text(1.0,1.3,'after one step',fontsize=8.5,color=GREEN); ax.set_xlabel('$x$',fontsize=9.5); ax.set_ylabel('$y$ and $\\mu$',fontsize=9.5); ax.set_ylim(0,6.5); save(fig,'p_14A3.png')
# 14A.4 multiplicative effects & exposure
b_,wy,wh=-2.5,0.5,0.8; mu_=0.5*np.exp(b_+wy+wh); A['14A.4']={'ratio_young':round(float(np.exp(wy)),4),'ratio_high':round(float(np.exp(wh)),4),'both':round(float(np.exp(wy+wh)),4),'rate':round(float(np.exp(b_+wy+wh)),4),'mu_half_year':round(float(mu_),4),'base_rate':round(float(np.exp(b_)),4)}
fig,ax=small(); lab=['base','young','high\npower','both']; val=[1,np.exp(wy),np.exp(wh),np.exp(wy+wh)]; ax.bar(range(4),val,color=['#C9D6EE',BLUE,BLUE,GREEN],edgecolor='white',width=0.65)
for i_,v in enumerate(val): ax.text(i_,v+0.1,f'×{v:.2f}' if i_ else '×1',ha='center',fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel('factor on $\\mu$',fontsize=9.5); ax.set_ylim(0,4.3); gr(ax); save(fig,'p_14A4.png')
# 14B.1 weights = repetition
xx5=np.arange(5.)[:,None]; yy5=np.array([1.,2,2,4,6]); w2=np.array([1,1,2,1,1.])
m1=PoissonRegressor(alpha=0.3,tol=1e-12,max_iter=500).fit(xx5,yy5,sample_weight=w2); idx=[0,1,2,2,3,4]; m2=PoissonRegressor(alpha=0.3,tol=1e-12,max_iter=500).fit(xx5[idx],yy5[idx])
A['14B.1']={'w_weighted':round(float(m1.coef_[0]),6),'w_repeated':round(float(m2.coef_[0]),6),'b_weighted':round(float(m1.intercept_),6),'b_repeated':round(float(m2.intercept_),6)}
fig,ax=small(); ax.scatter(np.arange(5.),yy5,s=30,color=BLUE,edgecolor='white',zorder=5); ax.scatter([2],[2],s=95,facecolor='none',edgecolor=OR,lw=1.5,zorder=4); ax.text(1.7,4.3,'weight 2\n= listed twice',fontsize=8.5,color=OR,va='center',ha='right'); gr(ax)
ax.set_xlabel('$x$',fontsize=9.5); ax.set_ylabel('$y$',fontsize=9.5); ax.set_ylim(0,7.5); save(fig,'p_14B1.png')
# 14B.2 gamma vs Poisson under rescaling
xg=np.arange(6.)[:,None]; yg=np.array([2.1,2.9,4.6,6.2,8.8,14.5]); res={}
for name,Sk in [('gamma',GammaRegressor),('poisson',PoissonRegressor)]:
    for a in (0.0,1.0):
        r1=Sk(alpha=a,tol=1e-12,max_iter=1000).fit(xg,yg); r2=Sk(alpha=a,tol=1e-12,max_iter=1000).fit(xg,1000*yg)
        res[f'{name}_alpha{a:g}']={'w':[round(float(r1.coef_[0]),4),round(float(r2.coef_[0]),4)],'b_diff':round(float(r2.intercept_-r1.intercept_),4)}
res['ln1000']=round(float(np.log(1000)),4); A['14B.2']=res
fig,ax=small(); xx=np.linspace(0,5.4,100); r1=GammaRegressor(alpha=1,tol=1e-12).fit(xg,yg); r2=GammaRegressor(alpha=1,tol=1e-12).fit(xg,1000*yg)
ax.plot(xx,r1.predict(xx[:,None]),color=GREEN,lw=1.7,label='$y$'); ax.plot(xx,r2.predict(xx[:,None])/1000,color=OR,lw=1.7,ls='--',label='$1000\\,y$ (÷1000)'); ax.scatter(xg,yg,s=26,color=BLUE,edgecolor='white',zorder=5)
ax.legend(fontsize=8.5,loc='upper left'); gr(ax); ax.set_xlabel('$x$',fontsize=9.5); ax.set_ylabel('$y$',fontsize=9.5); save(fig,'p_14B2.png')
# 14B.3 special cases
x6=xg; ya=yg
tw1=TweedieRegressor(power=1,alpha=0.2,tol=1e-12,max_iter=1000).fit(x6,ya); po=PoissonRegressor(alpha=0.2,tol=1e-12,max_iter=1000).fit(x6,ya)
tw2=TweedieRegressor(power=2,alpha=0.2,tol=1e-12,max_iter=1000).fit(x6,ya); ga=GammaRegressor(alpha=0.2,tol=1e-12,max_iter=1000).fit(x6,ya)
tw0=TweedieRegressor(power=0,alpha=0.2,tol=1e-12,max_iter=1000).fit(x6,ya); rd=Ridge(alpha=6*0.2).fit(x6,ya)
A['14B.3']={'p1_vs_poisson':float(abs(tw1.coef_[0]-po.coef_[0])),'p2_vs_gamma':float(abs(tw2.coef_[0]-ga.coef_[0])),'p0_vs_ridge':float(abs(tw0.coef_[0]-rd.coef_[0])),'w_poisson':round(float(po.coef_[0]),4),'w_gamma':round(float(ga.coef_[0]),4),'w_ridge':round(float(rd.coef_[0]),4)}
fig,ax=new(2.4,1.75)
for j,(a_,b_,c_) in enumerate([('power = 1','PoissonRegressor',BLUE),('power = 2','GammaRegressor',OR),('power = 0','Ridge($n\\alpha$)',GREEN)]):
    y_=1.2-j*0.5; box(ax,0,y_,1.0,0.36,a_,fc=LB,ec=c_,fs=8.5); box(ax,1.5,y_,1.35,0.36,b_,fc='#E3F4EA',ec=GREEN,fs=8.5); ax.text(1.25,y_+0.18,'=',ha='center',va='center',fontsize=12)
ax.text(1.4,-0.07,'TweedieRegressor on the left',fontsize=8.5,ha='center',color=GREY); ax.set_xlim(-0.05,2.95); ax.set_ylim(-0.1,1.7); save(fig,'p_14B3.png')
# 14C.1 log y regression vs gamma GLM
rng=np.random.default_rng(8); xs=rng.uniform(0,4,70); mu_t=np.exp(0.6+0.55*xs); ys=rng.gamma(1.2,mu_t/1.2)
gm=GammaRegressor(alpha=0,tol=1e-10,max_iter=500).fit(xs[:,None],ys); c=np.polyfit(xs,np.log(ys),1)
A['14C.1']={'gamma_at_4':round(float(gm.predict([[4.]])[0]),2),'logols_at_4':round(float(np.exp(c[1]+4*c[0])),2),'truth_at_4':round(float(np.exp(0.6+0.55*4)),2)}
fig,ax=small(); xx=np.linspace(0,4,100); ax.scatter(xs,ys,s=8,color=BLUE,alpha=0.8,zorder=3); ax.plot(xx,gm.predict(xx[:,None]),color=GREEN,lw=1.7,label='gamma GLM'); ax.plot(xx,np.exp(c[1]+c[0]*xx),color=OR,lw=1.7,ls='--',label='exp of fit to $\\ln y$'); gr(ax)
ax.legend(fontsize=8,loc='upper left'); ax.set_xlabel('$x$',fontsize=9.5); ax.set_ylabel('$y$',fontsize=9.5); ax.set_ylim(0,32); save(fig,'p_14C1.png')
# 14C.2 overdispersion
rng=np.random.default_rng(5); xo=rng.uniform(0,3,3000); mo=np.exp(0.3+0.6*xo); yp=rng.poisson(mo); r=4.0; ynb=rng.poisson(rng.gamma(r,mo/r))
bins=np.digitize(mo,np.linspace(1,6,6)); mm_=[];vp=[];vn=[]
for k in range(1,6):
    s_=bins==k
    if s_.sum()>20: mm_.append(mo[s_].mean()); vp.append(yp[s_].var()); vn.append(ynb[s_].var())
A['14C.2']={'mean':[round(float(v),2) for v in mm_],'var_poisson':[round(float(v),2) for v in vp],'var_overdispersed':[round(float(v),2) for v in vn]}
fig,ax=small(); ax.plot([0,8],[0,8],color=GREY,lw=1,ls=':'); ax.plot(mm_,vp,'o-',color=BLUE,ms=4,lw=1.3,label='Poisson'); ax.plot(mm_,vn,'s--',color=OR,ms=4,lw=1.3,label='overdispersed'); ax.legend(fontsize=8.5,loc='upper left'); gr(ax)
ax.set_xlabel('mean $\\mu$',fontsize=9.5); ax.set_ylabel('variance',fontsize=9.5); ax.set_xlim(0,6.5); ax.set_ylim(0,14); save(fig,'p_14C2.png')
# 14C.3 units and a silent stop
xk=np.arange(5.)[:,None]; yk=np.array([1.,2,2,4,6]); F=1e5; A['14C.3']={}
fig,axs=plt.subplots(1,2,figsize=(2.6,1.35),dpi=240,sharey=True); xx=np.linspace(-0.2,4.3,100)
for ax,(solver,sc_,lab) in zip(axs,[('lbfgs',F,'cycles, lbfgs'),('lbfgs',1.0,'$10^5$ cycles')]):
    m=PoissonRegressor(alpha=0.0,solver=solver,tol=1e-8,max_iter=1000).fit(sc_*xk,yk)
    A['14C.3'][lab]={'coef_x_scale':float(m.coef_[0]*sc_),'n_iter':int(m.n_iter_),'intercept':round(float(m.intercept_),4)}
    ax.plot(xx*sc_,m.predict(xx[:,None]*sc_),color=GREEN,lw=1.6); ax.scatter(xk*sc_,yk,s=18,color=BLUE,edgecolor='white',zorder=5); ax.set_xlabel(lab if sc_==F else '$x$ ($10^5$ cycles)',fontsize=8.5); ax.tick_params(labelsize=8); gr(ax)
axs[0].set_xlabel('$x$ (cycles)',fontsize=8.5); axs[0].set_xticks([0,2e5,4e5]); axs[0].set_xticklabels(['0','2e5','4e5'])
nc=PoissonRegressor(alpha=0.0,solver='newton-cholesky',tol=1e-8).fit(F*xk,yk); A['14C.3']['newton-cholesky, cycles']={'coef_x_scale':float(nc.coef_[0]*F),'n_iter':int(nc.n_iter_)}
axs[0].set_ylabel('$y$',fontsize=9.5); fig.tight_layout(w_pad=0.4); save(fig,'p_14C3.png')
json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

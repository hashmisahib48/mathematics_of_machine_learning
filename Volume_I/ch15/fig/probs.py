# Problem figures of Chapter 15. Run: python3 fig/probs.py  (answers written to verify/problem_answers.json)
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE+'/../../lib'); os.chdir(HERE+'/..')
from figstyle import *
from math import comb
from itertools import combinations_with_replacement as cwr
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
A={}
FS=9
def sm(w=2.5,h=1.5): return plt.subplots(figsize=(w,h),dpi=240)
def g(ax): ax.grid(color=GRID,lw=0.5); ax.set_axisbelow(True)
def sc(ax,xx,yy,s=26): ax.scatter(xx,yy,s=s,color=BLUE,edgecolor='white',lw=0.6,zorder=5)
plt.rcParams.update({'font.size':FS,'xtick.labelsize':FS,'ytick.labelsize':FS,'axes.labelsize':FS})

# ---- 15A.1 expand (3,-1)
p=PolynomialFeatures(2); z=p.fit_transform([[3.,-1.]])[0]; wv=np.array([1,2,-1,0.5,1,2.]); A['15A.1']={'columns':[float(v) for v in z],'yhat':float(z@wv),'count_p3_d3':comb(6,3),'check':int(PolynomialFeatures(3).fit(np.zeros((1,3))).n_output_features_)}
fig,ax=new(2.6,1.15); names=['1','$x_1$','$x_2$','$x_1^2$','$x_1x_2$','$x_2^2$']
for k,(n_,v) in enumerate(zip(names,z)):
    ax.add_patch(Rectangle((k*0.42,0.2),0.4,0.38,fc=LB,ec=BLUE,lw=0.8)); ax.text(k*0.42+0.2,0.39,f'{v:g}'.replace('-','−'),ha='center',va='center',fontsize=FS,color=NAVY); ax.text(k*0.42+0.2,0.72,n_,ha='center',va='bottom',fontsize=FS,color=BLUE)
ax.text(1.26,1.18,'sample $(x_1, x_2) = (3, {-1})$',ha='center',fontsize=FS,color=GREY); ax.set_xlim(-0.05,2.6); ax.set_ylim(0.05,1.3); save(fig,'p_15A1.png')

# ---- 15A.2 quadratic by orthogonal columns
x=np.array([1.,2,3,4,5]); y=np.array([6.,2,1,4,12]); u=x-3; q=u**2-2; c1=u@y/(u@u); c2=q@y/(q@q); fit=y.mean()+c1*u+c2*q
w3=np.linalg.lstsq(np.c_[np.ones(5),x,x**2],y,rcond=None)[0]
A['15A.2']={'ybar':float(y.mean()),'c1':round(float(c1),4),'c2':round(float(c2),4),'raw':[round(float(v),4) for v in w3],'SSE':round(float(((y-fit)**2).sum()),4),'at6':round(float(w3@[1,6,36]),3)}
fig,ax=sm(); sc(ax,x,y,30); xx=np.linspace(0.6,5.4,100); ax.plot(xx,w3[0]+w3[1]*xx+w3[2]*xx**2,color=GREEN,lw=1.6); g(ax); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); ax.set_ylim(-1,14); save(fig,'p_15A2.png')

# ---- 15A.3 cubic B-spline values
s=SplineTransformer(knots=np.array([[0.],[1],[2],[3],[4]]),degree=3).fit(np.zeros((2,1))); tt=np.linspace(0,4,300)[:,None]
B=s.transform(tt); col=3  # function on knots 0..4
vals=s.transform([[1.],[2.],[3.]])[:,col]; row2=s.transform([[2.]])[0]
A['15A.3']={'B_at_1_2_3':[round(float(v),4) for v in vals],'row_at_2':[round(float(v),4) for v in row2],'quad_at_2':[0.5,0.5]}
fig,ax=sm(); ax.plot(tt[:,0],B[:,col],color=PUR,lw=1.8); ax.plot([2],[2/3],'o',color=RED,ms=6,zorder=5); ax.set_xticks([0,1,2,3,4]); ax.set_ylim(0,0.8); g(ax); ax.set_xlabel('$x$  (knots 0, 1, 2, 3, 4)'); ax.set_ylabel('$B(x)$'); ax.text(0.15,0.55,'$B(2)=\\,?$',color=RED,fontsize=FS+1); save(fig,'p_15A3.png')

# ---- 15A.4 counting
pc=6; A['15A.4']={'poly_p6_d4':comb(10,4),'interaction_p6_d3':sum(comb(6,k) for k in range(4)),'spline_4feat':4*(7+3-1-1)}
assert PolynomialFeatures(4).fit(np.zeros((1,6))).n_output_features_==comb(10,4); assert PolynomialFeatures(3,interaction_only=True).fit(np.zeros((1,6))).n_output_features_==A['15A.4']['interaction_p6_d3']
assert SplineTransformer(n_knots=7,degree=3,include_bias=False).fit(np.zeros((8,4))).n_features_out_==32
fig,ax=sm(); ds=np.arange(1,9)
for pp,c,ls in ((2,BLUE,'-'),(6,GREEN,'--'),(10,RED,':')): ax.semilogy(ds,[comb(pp+d,d) for d in ds],ls,color=c,lw=1.6,marker='o',ms=2.5,label=f'$p$ = {pp}')
ax.legend(fontsize=FS-1,loc='upper left',ncol=3,handlelength=1.4,columnspacing=0.6); ax.set_ylim(1,3e7); g(ax); ax.set_xlabel('degree $d$'); ax.set_ylabel('columns'); save(fig,'p_15A4.png')

# ---- 15B.1 interaction_only in the scratch class
class MyPoly:
    def __init__(s,degree=2,interaction_only=False): s.degree,s.io=degree,interaction_only
    def fit(s,X,y=None): s.p_=np.asarray(X).shape[1]; return s
    def transform(s,X):
        X=np.asarray(X,float); cols=[]
        for k in range(s.degree+1):
            for c in cwr(range(s.p_),k):
                if s.io and len(set(c))<len(c): continue
                cols.append(np.prod(X[:,list(c)],axis=1))
        return np.column_stack(cols)
Xr=np.random.default_rng(0).normal(size=(6,4)); a_=MyPoly(3,True).fit(Xr).transform(Xr); b_=PolynomialFeatures(3,interaction_only=True).fit_transform(Xr)
A['15B.1']={'agree':bool(a_.shape==b_.shape and np.allclose(a_,b_)),'columns':int(a_.shape[1])}
fig,ax=new(2.6,1.2)
for k,(t_,c,fc) in enumerate((('degree 3\n$p$ = 4',BLUE,LB),('all\nproducts',GREY,LGREY),('skip\nrepeats',GREEN,'#E3F4EA'))):
    box(ax,k*0.88,0.2,0.8,0.8,t_,fc=fc,ec=c,fs=FS)
arr(ax,(0.8,0.6),(0.88,0.6),c=NAVY); arr(ax,(1.68,0.6),(1.76,0.6),c=NAVY); ax.set_xlim(-0.05,2.6); ax.set_ylim(0.1,1.1); save(fig,'p_15B1.png')

# ---- 15B.2 uniform vs quantile knots on skewed data
def curve(xx): return 4*np.exp(-(xx-3)**2/0.8)+0.5*xx
rng=np.random.default_rng(0); xs=np.sort(rng.exponential(2.5,80)); xs=xs[xs<10][:,None]; ys=curve(xs[:,0])+rng.normal(0,0.4,len(xs))
xn=rng.exponential(2.5,3000); xn=xn[xn<10][:,None]; yn=curve(xn[:,0])+rng.normal(0,0.4,len(xn)); ks=list(range(4,17)); res={}
for kind in ('uniform','quantile'):
    r=[]
    for k in ks:
        m=make_pipeline(SplineTransformer(n_knots=k,knots=kind,include_bias=False),LinearRegression()).fit(xs,ys); r.append(float(np.sqrt(np.mean((m.predict(xn)-yn)**2))))
    res[kind]=r
A['15B.2']={'n_train':int(len(xs)),'uniform_best':[ks[int(np.argmin(res['uniform']))],round(min(res['uniform']),3)],'quantile_best':[ks[int(np.argmin(res['quantile']))],round(min(res['quantile']),3)],'uniform_k16':round(res['uniform'][-1],2),'quantile_k16':round(res['quantile'][-1],3)}
fig,ax=sm(); ax.plot(ks,res['uniform'],'--',color=BLUE,lw=1.5,label='uniform'); ax.plot(ks,res['quantile'],'-',color=GREEN,lw=1.6,label='quantile'); ax.set_ylim(0.3,1.6); ax.legend(fontsize=FS-0.5,loc='upper left',handlelength=2.0); g(ax); ax.set_xlabel('n_knots'); ax.set_ylabel('test RMSE'); save(fig,'p_15B2.png')

# ---- 15B.3 tol and the rank of raw polynomial columns
yr=np.arange(1990,2020.)[:,None]; t_=yr[:,0]-1990; yy=5+0.3*t_+0.02*t_**2+np.random.default_rng(0).normal(0,0.5,30); tols=[1e-6,1e-9,1e-12,1e-15]; out={}
for d in (2,3,4):
    out[str(d)]=[]
    for tol in tols:
        m=make_pipeline(PolynomialFeatures(d,include_bias=False),LinearRegression(tol=tol)).fit(yr,yy); out[str(d)].append([int(m[-1].rank_),round(float(m.score(yr,yy)),4)])
A['15B.3']={'tols':tols,'rank_r2':out}
fig,ax=sm()
for d,c,ls in ((2,BLUE,'-'),(3,GREEN,'--'),(4,RED,':')): ax.semilogx(tols,[r[1] for r in out[str(d)]],ls,color=c,lw=1.6,marker='o',ms=3,label=f'degree {d}')
ax.invert_xaxis(); ax.legend(fontsize=FS-0.5,loc='lower right',handlelength=2.0); g(ax); ax.set_xlabel('tol'); ax.set_ylabel('training $R^2$'); save(fig,'p_15B3.png')

# ---- 15C.1 columns against samples
n=200; p_=30; mcols=comb(p_+3,3); A['15C.1']={'columns':mcols,'n':n}
fig,ax=sm(2.5,1.3); ax.bar(['samples\n$n$','columns\n$m$'],[n,mcols],color=[BLUE,PUR],width=0.55); ax.set_ylabel('count'); g(ax); ax.set_ylim(0,7000)
for k,v in enumerate((n,mcols)): ax.text(k,v+120,f'{v}',ha='center',fontsize=FS,color=NAVY)
save(fig,'p_15C1.png')

# ---- 15C.2 knots on skewed data
fig,ax=sm(2.5,1.3); ax.hist(xs[:,0],bins=14,color=LGREY,edgecolor='white'); ku=np.linspace(xs.min(),xs.max(),7); kq=np.quantile(xs[:,0],np.linspace(0,1,7))
ax.plot(ku,np.full(7,-1.6),'|',color=BLUE,ms=7,mew=1.6,clip_on=False,label='uniform'); ax.plot(kq,np.full(7,-3.4),'|',color=GREEN,ms=7,mew=1.6,clip_on=False,label='quantile')
ax.set_xlabel('$x$',labelpad=14); ax.set_ylabel('count'); ax.legend(fontsize=FS-0.5,loc='upper right',handlelength=1.0); save(fig,'p_15C2.png')

# ---- 15C.3 sine and cosine against periodic spline
def circ(h,c): d=np.abs(h-c); return np.minimum(d,24-d)
def demand(h): return 2+5*np.exp(-circ(h,8)**2/4)+3*np.exp(-circ(h,17.5)**2/6)
hh=np.linspace(0,24,200); Z=np.c_[np.sin(2*np.pi*hh/24),np.cos(2*np.pi*hh/24)]; f1=LinearRegression().fit(Z,demand(hh)); sp=SplineTransformer(knots=np.linspace(0,24,9)[:,None],extrapolation='periodic'); f2=LinearRegression().fit(sp.fit_transform(hh[:,None]),demand(hh))
fig,ax=sm(2.5,1.3); ax.plot(hh,demand(hh),color=GREY,lw=1.0,ls=':'); ax.plot(hh,f1.predict(Z),color=OR,lw=1.5,ls='--',label='sin, cos'); ax.plot(hh,f2.predict(sp.transform(hh[:,None])),color=GREEN,lw=1.5,label='periodic spline')
ax.set_xticks([0,12,24]); ax.legend(fontsize=FS-0.5,loc='upper center',ncol=2,handlelength=1.8,columnspacing=1.0); g(ax); ax.set_xlabel('hour'); ax.set_ylabel('demand'); ax.set_ylim(0,10.5); save(fig,'p_15C3.png')

json.dump(A,open('verify/problem_answers.json','w'),indent=1); print(json.dumps(A))

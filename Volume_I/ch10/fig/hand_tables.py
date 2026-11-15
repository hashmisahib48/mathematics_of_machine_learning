# Numbers for the worked examples of Chapter 10 -> verify/hand_tables.json
import os,json,numpy as np; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..')
x=np.array([1.,2,3,4,5]); y=np.array([3.,4,6,7,10]); x2=np.array([1.,2,3,5,4]); T={}
xc=x-3; yc=y-6
for a in (0,10):
    w=17/(10+a); r=yc-w*xc; T[f'one_a{a}']={'w':w,'b':6-3*w,'fit':list(np.round(6+w*xc,4)),'r':list(np.round(r,4)),'r2':list(np.round(r**2,4)),'SSE':round(float(r@r),4),'R2':round(1-float(r@r)/30,4),'pred6':round(6-3*w+6*w,4)}
for dw in (0.8,0.9,0.85): T[f'J_a10_w{dw}']=round(30-34*dw+20*dw**2,4)
X=np.c_[x,x2]; Xc=X-X.mean(0)
for a in (0,1):
    w=np.linalg.solve(Xc.T@Xc+a*np.eye(2),Xc.T@yc); b=6-3*w.sum(); fit=b+X@w; r=y-fit
    T[f'two_a{a}']={'w':list(np.round(w,4)),'b':round(float(b),4),'fit':list(np.round(fit,3)),'r':list(np.round(r,3)),'SSE':round(float(r@r),4),'pred66':round(float(b+6*w.sum()),3),'sum_w':round(float(w.sum()),4)}
# variances along v (sigma^2 = 1)
T['var']={'ols':[1/19,1/1],'ridge1':[19/400,1/4]}
g=np.array([17.,14.]); v=[np.array([1,1])/np.sqrt(2),np.array([1,-1])/np.sqrt(2)]; lam=[19.,1.]
T['eig']={'vg':[round(float(vv@g),4) for vv in v],'c_ols':[round(float(vv@g/l),4) for vv,l in zip(v,lam)],'c_ridge1':[round(float(vv@g/(l+1)),4) for vv,l in zip(v,lam)]}
json.dump(T,open('verify/hand_tables.json','w'),indent=1); print(json.dumps(T))

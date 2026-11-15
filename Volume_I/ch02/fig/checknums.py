# Checks every hand-worked number of Chapter 2 and writes verify/numbers.json
import numpy as np, json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..'))
N={}; ok=np.allclose
u=np.array([3.,1]); v=np.array([1.,2])
assert ok(u+v,[4,3]) and ok(2*u-v,[5,0]); assert u@v==5
cos=u@v/np.linalg.norm(u)/np.linalg.norm(v); assert ok(cos,2**-.5) and ok(np.degrees(np.arccos(cos)),45)
assert u@np.array([-1,3.])==0
w=np.array([3.,-4,12]); assert [np.linalg.norm(w,q) for q in (2,1,np.inf)]==[13,19,12]
c=(u@v)/(u@u); p=c*u; assert c==0.5 and ok(p,[1.5,.5]) and ok(v-p,[-.5,1.5]) and ok(u@(v-p),0)
A=np.array([[2.,1],[1,0],[0,1]]); b=np.array([3.,3,3])
assert ok(A.T@A,[[5,2],[2,2]]) and ok(A.T@b,[9,6]) and ok(np.linalg.det(A.T@A),6)
cc=np.linalg.solve(A.T@A,A.T@b); assert ok(cc,[1,2]); pp=A@cc; r=b-pp
assert ok(pp,[4,1,2]) and ok(r,[-1,2,1]) and ok(A.T@r,0) and ok(b@b,27) and ok(pp@pp+r@r,27) and pp@pp==21 and r@r==6
M=np.array([[1.,2],[3,4]]); Nn=np.array([[2.,0],[1,3]])
assert ok(M@Nn,[[4,6],[10,12]]) and ok(Nn@M,[[2,4],[10,14]]) and ok(M@[1,1],[3,7]) and ok((M@Nn).T,Nn.T@M.T) and ok(Nn.T@M.T,[[4,10],[6,12]])
K=np.array([[4.,2],[3,2]]); assert ok(np.linalg.det(K),2) and ok(np.linalg.inv(K),[[1,-1],[-1.5,2]]) and ok(np.linalg.solve(K,[10,8]),[2,1])
S0=np.array([[2.,4],[1,2]]); assert ok(np.linalg.det(S0),0)
a1,a2=A[:,0],A[:,1]; q1=a1/np.linalg.norm(a1); v2=a2-(q1@a2)*q1
assert ok(q1@a2*q1,[.8,.4,0]) and ok(v2,[.2,-.4,1]) and ok(np.linalg.norm(v2),1.2**.5) and ok(v2/np.linalg.norm(v2),[0.1826,-0.3651,0.9129],atol=1e-4)
S=np.array([[5.,2],[2,2]]); assert np.trace(S)==7 and ok(np.linalg.det(S),6) and ok(np.sort(np.linalg.eigvalsh(S)),[1,6])
assert ok(S@[2,1],6*np.array([2,1])) and ok(S@[1,-2],[1,-2])
s=np.linalg.svd(A,compute_uv=False); assert ok(s,[6**.5,1]); assert ok(A.T@A,S)
u1=A@np.array([2,1])/5**.5/6**.5; u2=A@np.array([1,-2])/5**.5; assert ok(u1,np.array([5,2,1])/30**.5) and ok(u2,np.array([0,1,-2])/5**.5) and ok(u1@u2,0)
a=np.array([[2.,1],[2,1.1]]).T; assert ok(np.linalg.solve(a,[4,5]),[-28,30])
assert ok(np.linalg.solve(np.array([[2.,1],[1,2]]),[4,5]),[1,2])
d=[np.linalg.det(m) for m in (np.diag([2,.75]),np.array([[1,1],[0,1.]]),np.array([[.5,.5],[.5,.5]]))]; assert ok(d,[1.5,1,0])
assert ok(np.linalg.solve(np.array([[3.,1],[5,2]]),[1,2]),[0,1]) and ok(np.linalg.det([[3,1],[5,2]]),1)
assert ok(np.sort(np.linalg.eigvalsh([[3,1],[1,3.]])),[2,4])
N['ok']=True; json.dump(N,open('verify/numbers.json','w')); print('all hand numbers verified')

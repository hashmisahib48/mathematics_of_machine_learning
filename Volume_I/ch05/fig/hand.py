# Hand-example numbers for Chapter 5 -> verify/hand.json
import os,json; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..')
import numpy as np
x=np.arange(1.,6); y=np.array([3.,4,6,7,10]); X=np.c_[np.ones(5),x]; H=2*X.T@X
J=lambda v:float(((y-X@v)**2).sum()); g=lambda v:-2*X.T@(y-X@v)
A={}
# Example 5.3-1: f(w)=2(w-5)^2 from 0
f=lambda w:2*(w-5)**2
for eta in (0.1,0.3,0.5):
    w=0.;P=[w]
    for _ in range(4): w=w-eta*4*(w-5);P.append(round(w,4))
    A[f'quad_eta{eta}']=P
# Example 5.3-2: delivery, eta=0.01, two steps from 0
v=np.zeros(2); S=[]
for t in range(3):
    S.append({'w':v.round(4).tolist(),'grad':g(v).round(4).tolist(),'J':round(J(v),4)}); v=v-0.01*g(v)
S.append({'w':v.round(4).tolist(),'grad':g(v).round(4).tolist(),'J':round(J(v),4)})
A['delivery_gd']=S
A['eig']=np.linalg.eigvalsh(H).round(2).tolist(); A['limit']=round(2/np.linalg.eigvalsh(H)[-1],5)
A['H']=H.tolist(); A['Hinv_times_200']=(np.linalg.inv(H)*200).round(4).tolist()
# Section 5.7: dual of  min (w-3)^2  s.t.  w <= 1  (and the KKT example)
from scipy.optimize import minimize_scalar
q=lambda lam: minimize_scalar(lambda w:(w-3)**2+lam*(w-1)).fun   # dual function, by numbers
lam_best=minimize_scalar(lambda l:-q(l),bounds=(0,20),method='bounded').x
A['dual']={'q_formula_at_2':2*2-2**2/4,'q_numeric_at_2':round(q(2.0),6),'lam_star':round(lam_best,4),'q_star':round(q(lam_best),6),
 'primal_min':(1-3)**2,'gap_w0_lam2':round((0-3)**2-q(2.0),6)}
json.dump(A,open('verify/hand.json','w'),indent=1)
print(json.dumps(A))

# Numbers quoted in the text of Chapter 15 -> verify/numbers.json. Run: python3 fig/numbers.py
import sys,os,json; HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE+'/..')
import numpy as np
from math import comb
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LinearRegression
N={}
tab={}
for p in (1,2,5,10,50):
    for d in (2,3,5,10):
        m=comb(p+d,d); io=sum(comb(p,k) for k in range(0,min(d,p)+1)); tab[f'{p},{d}']=[m,io]
        if m<=30000:
            pf=PolynomialFeatures(d).fit(np.zeros((1,p))); assert pf.n_output_features_==m
            pi=PolynomialFeatures(d,interaction_only=True).fit(np.zeros((1,p))); assert pi.n_output_features_==io
N['counts']=tab
# quadratic B-spline at 1.5, knots 0..3
s=SplineTransformer(knots=np.array([[0.],[1],[2],[3]]),degree=2).fit(np.zeros((2,1)))
N['quad_bspline_at_1.5']=[round(float(v),4) for v in s.transform([[1.5]])[0]]
N['quad_bspline_at_1.0']=[round(float(v),4) for v in s.transform([[1.0]])[0]]
# toy: extrapolated predictions at x=6
x=np.array([1.,2,3,4,5]); y=np.array([2.,3,8,15,22])
N['toy_line_at_6']=round(float(LinearRegression().fit(x[:,None],y).predict([[6.]])[0]),2)
H=SplineTransformer(n_knots=3,degree=1).fit_transform(x[:,None]); v=np.linalg.solve(H.T@H,H.T@y)
N['hat_w']=[round(float(a),4) for a in v]; N['hat_w_fractions_check']=[46/35,52/7,774/35]
N['hat_pred_at_6']=round(float((SplineTransformer(n_knots=3,degree=1).fit(x[:,None]).transform([[6.]])@v)[0]),3)
N['hat_XtX']=(H.T@H).tolist(); N['hat_Xty']=(H.T@y).tolist()
# column sums of orthogonality check
u=x-x.mean(); q=u**2-np.mean(u**2); N['orth']=[float(u.sum()),float(q.sum()),float(u@q)]; N['q']=q.tolist(); N['uy']=float(u@y); N['qy']=float(q@y); N['uu']=float(u@u); N['qq']=float(q@q)
# expansion example: x=(2,3)
N['expand_2_3']=PolynomialFeatures(2).fit_transform([[2.,3.]]).tolist()[0]
# degree-3 p=2 names
N['poly_p2_d3_cols']=int(PolynomialFeatures(3).fit(np.zeros((1,2))).n_output_features_)
N['toy_quad_slope']=[round(-0.8+2*1.0*xx,1) for xx in (1,5)]
N['toy_quad_at_6']=round(float(np.array([1.4,-0.8,1.0])@[1,6,36]),1)
N['mem_p50_d3_n10000_GB']=round(10000*comb(53,3)*8/1e9,2)
json.dump(N,open('verify/numbers.json','w'),indent=1); print(json.dumps(N)[:1500])

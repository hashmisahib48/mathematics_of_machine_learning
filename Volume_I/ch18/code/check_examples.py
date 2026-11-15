# Verification of the numbers printed in the worked examples of Section 18.6 (not a book listing).
import numpy as np
z=lambda x,b:4*x+b
b_data=-20-np.log(1.5); b_eq=-20.0
print('b_data',round(b_data,4),'b_eq',b_eq)
for x in (4,5,5.05,6): print(x,round(z(x,b_data),4),round(1/(1+np.exp(-z(x,b_data))),4))
print('equal priors z at 5.05',round(z(5.05,b_eq),4),round(1/(1+np.exp(-z(5.05,b_eq))),4))
v0,v1=8/6,2/4
a2=1/(2*v0)-1/(2*v1); a1=7/v1-3/v0; a0=9/(2*v0)-49/(2*v1)-0.5*np.log(v1/v0)+np.log(0.4/0.6)
print('a',a2,a1,a0,'terms',9/(2*v0),49/(2*v1),-0.5*np.log(v1/v0),np.log(0.4/0.6))
disc=a1**2-4*a2*a0; print('disc',disc,np.sqrt(disc)); print('roots',(-a1+np.sqrt(disc))/(2*a2),(-a1-np.sqrt(disc))/(2*a2))
print('x=14 dist',(14-7)**2/v1,(14-3)**2/v0)
print('ratio n/(n-K) wine',178/175)
print('ln1.5',np.log(1.5),'0.25ln1.5',0.25*np.log(1.5))
print('sigma(1.6)',1/(1+np.exp(-1.6)))

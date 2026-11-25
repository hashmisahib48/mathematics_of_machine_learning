xt=[-2,-1,0,1,2]; yt=[-10,-5,-1,6,10]
w=0.0
for k in range(3):
    g=-2/5*sum(a*(b-w*a) for a,b in zip(xt,yt)); J=sum((b-w*a)**2 for a,b in zip(xt,yt))/5
    print('iter',k+1,'w',round(w,4),'J',round(J,4),'g',round(g,4)); w=w-0.1*g
print('w3',round(w,4))
w=0.0
for a,b in zip(xt,yt):
    e=b-w*a; w=w+2*0.02*a*e; print('sgd e',round(e,4),'w',round(w,5))
w=0.0
for k in range(3):
    g=-2/5*sum(a*(b-w*a) for a,b in zip(xt,yt)); w=w-0.6*g; print('eta0.6 w',round(w,4),'err',round(w-5.1,4))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:41:00 2019

@author: shtabari
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


data0=np.loadtxt("data.txt",skiprows=19)
#np.random.shuffle(data0)
###################################### scaling and centring 1.1
colmean=np.min(data0,axis=0)
datac=data0-colmean
data = datac / (np.ptp(datac,axis=0))

datatr,datats=data[:48,:], data[48:,:]
md=np.size(data,axis=0)
m=np.size(datatr,axis=0)
n=np.size(datatr,axis=1)-1


one=np.ones(m).reshape((m,1))
y=datatr[:,n].reshape((m,1))
XX=datatr[:,1:16].reshape((m,n-1))

X=np.hstack((one,XX))

def dot2(a,b):
    n=np.size(a,axis=0)
    m=np.size(b,axis=1)
    return np.dot(a,b).reshape(n,m)

def dot3(a,b,c):
    return np.dot(np.dot(a,b),c)
def dot4(a,b,c,d):
    return np.dot( np.dot(a,b), np.dot(c,d))
def inv(a):
    return np.linalg.inv(a)
def inv2(a,b):
    return np.linalg.inv(np.dot(a,b))

beta_least= dot3(inv2(X.T,X),X.T,y)

lam=1
alpha=0.01
epsi=0.1
init=(0+np.zeros(n)).reshape(n,1)
#init=(10*np.random.randn(n)).reshape(n,1)

IP=np.identity(n)
#RI=inv2(X.T,X)

beta_ridge=dot2(inv((lam*IP)+dot2(X.T,X)),dot2(X.T,y))


def loss_fr(beta,X,y,lam):
    m=len(y)
    PT=dot2(beta.T,X.T)
    eT= y.T - PT
    e2= dot2(eT,eT.T)
    J=(e2/2/m) + ((lam/2/m)*dot2(beta.T,beta))
    return J

def err2r(beta,X,y):
    m=len(y)
    PT=dot2(beta.T,X.T)
    eT= y.T - PT
    e2= dot2(eT,eT.T)
    return e2/(2*m)



def rigid(X,y,beta):
    m=len(y)    
    delta=[]
    betah=[] ### betas' history
    J0=[]
    ERR=[]
    k=1
    d=1
    while d >= epsi:
        Jold=loss_fr(beta,X,y,lam)
        beta = beta - ((1/m)*alpha*(dot2(X.T,( (dot2(X,beta)) - y)))) \
                - (alpha*lam/m)*beta
        Jnew=loss_fr(beta,X,y,lam)
        
        d=(100*np.abs(Jold-Jnew)/Jold)
        delta.append(d)
        betah.append(beta)
        J0.append(loss_fr(beta,X,y,lam))
        ERR.append(err2r(beta,X,y))
        k += 1
#    return beta,loss_history,loss_history
    return beta,betah,J0,ERR,delta,k

#init=(10*np.random.randn(n)).reshape(n,1)

rigid=rigid(X,y,init)
lost=[i[0][0] for i in rigid[2]]
kk=[i for i in range(rigid[-1])]
     
fig, ax = plt.subplots()
ax.plot(lost)
ax.set(xlabel='k # of iterations', ylabel='Loss Function')
fig.savefig("1.2.eps",format="eps")
plt.show()


betasr = [x*0.0 if x < 0.005 else x for x in rigid[0]]
print(n-betasr.count(0))
################################# 
one=np.ones(md-m).reshape((md-m,1))
yts=datats[:,n].reshape((md-m,1))

XXts=datats[:,1:16].reshape((md-m,n-1))
Xts=np.hstack((one,XXts))
##################################

SL_TS=err2r(np.asarray(rigid[0]),Xts,yts)
print(SL_TS)

#print(dot2(d.T,d))
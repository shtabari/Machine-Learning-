#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 18:41:00 2019

@author: shtabari
"""


import numpy as np
import matplotlib.pyplot as plt

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

lam=1
alpha=0.01
epsi=0.1

IP=np.identity(n)
init=(0+np.zeros(n)).reshape(n,1)


def loss_fl(beta,X,y,lam):
    m=len(y)
    PT=dot2(beta.T,X.T)
    eT= y.T - PT
    e2= dot2(eT,eT.T)
    J=(e2/2/m) + ((lam/2/m)*np.sum(np.abs(beta)))
    return J

def err2l(beta,X,y):
    m=len(y)
    PT=dot2(beta.T,X.T)
    eT= y.T - PT
    e2= dot2(eT,eT.T)
    return e2/(2*m)
    
def dl1(beta):
    dl1=(np.asarray([1 if i >= 0 else -1 for i in beta]))
    return dl1.reshape(np.size(beta),1)


def lasso(X,y,beta):
    m=len(y)    
    delta=[]
    betah=[] ### betas' history
    J0=[]
    ERR=[]
    k=1
    d=1
    while d > epsi:
        Jold=loss_fl(beta,X,y,lam)
        beta = beta - ((1/m)*alpha*(dot2(X.T,( (dot2(X,beta)) - y)))) \
                - ((alpha*lam/m/2)*dl1(beta))
        Jnew=loss_fl(beta,X,y,lam)
        d=(100*np.abs(Jold-Jnew)/Jold)
        delta.append(d)
        betah.append(beta)
        J0.append(loss_fl(beta,X,y,lam))
        ERR.append(err2l(beta,X,y))
        k += 1
#    return beta,loss_history,loss_history
    return beta,betah,J0,ERR,delta,k



lasso=lasso(X,y,init)

lost=[i[0][0] for i in lasso[2]]
plt.plot(lost)
#plt.plot(lasso[6])



betasl = [x*0.0 if x < 0.005 else x for x in lasso[0]]
print(n-betasl.count(0))
################################# 
one=np.ones(md-m).reshape((md-m,1))
yts=datats[:,1].reshape((md-m,1))
XXts=datats[:,2::].reshape((md-m,n-1))
Xts=np.hstack((one,XXts))
##################################

SL_TS=err2l(np.asarray(betasl),Xts,yts)
print(SL_TS)

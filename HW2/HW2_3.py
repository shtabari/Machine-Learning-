#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 10:12:16 2019

@author: shtabari
"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
############################################# READINF DATA

data0=np.loadtxt("data.txt",skiprows=13)
np.random.shuffle(data0)
m=np.size(data0,axis=0)
n=np.size(data0,axis=1) - 1
y=data0[:,n].reshape((m,1))
X=data0[:,0:n].reshape((m,n))

w=np.zeros((n,1))
b=np.zeros((1,1))
C=10
bs=4
############################################# auxiliary functions

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

############################################# Mini Batch Gradient Descent Cost Function

def cost_mb(C,X,y,w,b):
    wsum=0.5 * dot2(w.T,w)
    wxb=(dot2(X,w)) + b
    ywxb=1 - (y * wxb)
    cmax=(C*sum([0 if i<0 else i for i in ywxb]))    
    return wsum+cmax
############################################# gradient of the function

def grad_mb(C,X,y,w,b,ii):
    n=np.size(w)
    m=np.size(y)
    lw=np.zeros(n)
    cond=(y * dot2(X,w)) + b
    dw=(ii*bs)
    up=(min(m,(ii+1)*bs))
    db=sum([-y[i] if cond[i]<1 else 0 for i in range(dw,up)])
    for j in range(n):
        condd=[]
        for i in range(dw,up):
            if cond[i] < 1:
                condd.append(-1*y[i]*X[i,j])
        lw[j]=sum(condd)  
    return (np.squeeze(w)+(C*lw)).reshape(n,1),(C*db)
############################################# MB_SVM Engine 

def svm_mb(X,y,w,b,eta=0.00000001,eps=0.004,C=10):
    m=np.size(y)
    k=0
    ii=0 ### ii it is l 
    dd=1
    dlist=[]
    clist = []
    dlist.append(0)
    clist.append(cost_mb(C,X,y,w,b))
    while dd > eps:
        w = w - eta*(grad_mb(C,X,y,w,b,ii)[0])
        b = b - eta*(grad_mb(C,X,y,w,b,ii)[1])
        k = k + 1
#        ii = int(((ii+1) % ((m+bs)/bs)))
        ii = (ii+1) % int((m+bs)/bs)
        clist.append(cost_mb(C,X,y,w,b))
        dlist.append(0.5*dlist[k-1]+0.5*(100*abs(clist[k-1]-clist[k])/clist[k-1]))
        dd=dlist[k]
    print(k)
    return(clist)
############################################# Running and Ploting 

svm_mb=svm_mb(X,y,w,b)
svm_mb=[i[0][0] for i in svm_mb]
fig, ax = plt.subplots()
ax.plot(svm_mb)
ax.set(xlabel='k # of iterations', ylabel='Loss Function',title='Mini Batch Gradient Descent')
fig.savefig("HW2_3.eps",format="eps")
plt.show()
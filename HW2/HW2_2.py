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

############################################# Stochastic Gradient Descent Cost Function
def cost_s(C,X,y,w,b):
    wsum=0.5 * dot2(w.T,w)
    wxb=(dot2(X,w)) + b
    ywxb=1 - (y * wxb)
    cmax=(C*sum([0 if i<0 else i for i in ywxb]))    
    return wsum+cmax
############################################# gradient of the function

def grad_s(C,X,y,w,b,ii):
    n=np.size(w)
    m=np.size(y)
    lw=np.zeros(n)
    cond=(y * dot2(X,w)) + b
    db=(-1*y[ii]) if cond[ii] < 1 else 0
    for j in range(n):
        if cond[ii] < 1:
            lw[j]=-1*y[ii]*X[ii,j]
        else:
            lw[j]=0       
    return (np.squeeze(w)+(C*lw)).reshape(n,1),(C*db)
############################################# S_SVM Engine 
def svm_s(X,y,w,b,eta=0.00000001,eps=0.0003,C=10):
    m=np.size(y)
    k=0
    ii=0
    dd=1
    dlist=[]
    clist = []
    dlist.append(0)
    clist.append(cost_s(C,X,y,w,b))
    while dd > eps:
        w = w - eta*(grad_s(C,X,y,w,b,ii)[0])
        b = b - eta*(grad_s(C,X,y,w,b,ii)[1])
        k = k + 1
        ii = (ii+1) % m
        clist.append(cost_s(C,X,y,w,b))
        dlist.append(0.5*dlist[k-1]+0.5*(100*abs(clist[k-1]-clist[k])/clist[k-1]))
        dd=dlist[k]
    print(k)
    return(clist)
############################################# Running and Ploting 
svm_s=svm_s(X,y,w,b)
svm_s=[i[0][0] for i in svm_s]
fig, ax = plt.subplots()
ax.plot(svm_s)
ax.set(xlabel='k # of iterations', ylabel='Loss Function',title='Stochastic Gradient Descent')
fig.savefig("HW2_2.eps",format="eps")
plt.show()



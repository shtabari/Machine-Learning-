#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:40:30 2019

@author: shtabari
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
############################################# READINF DATA
data0=np.loadtxt("data.txt",skiprows=13)
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

############################################# Batch Gradient Descent Cost Function
def cost_b(C,X,y,w,b):
    wsum=0.5 * dot2(w.T,w)
    wxb=(dot2(X,w)) + b
    ywxb=1 - (y * wxb)
    cmax=(C*sum([0 if i<0 else i for i in ywxb]))    
    return wsum+cmax
############################################# gradient of the function
def grad_b(C,X,y,w,b):
    n=np.size(w)
    m=np.size(y)
    lw=np.zeros(n)
    cond=(y * dot2(X,w)) + b
    db=sum([-y[i] if cond[i]<1 else 0 for i in range(m)])
    for j in range(n):
        condd=[]
        for i in range(m):
            if cond[i] < 1:
                condd.append(-1*y[i]*X[i,j])
        lw[j]=sum(condd)
            
    return (np.squeeze(w)+(C*lw)).reshape(n,1),(C*db)
#    return n

############################################# SVM Engine 
def svm_b(X,y,w,b,eta=0.000000001,eps=0.04,C=10):
    k=0
    cost_list=[]
    d=1
    while d > eps:
        costold=cost_b(C,X,y,w,b)
        w = w - eta*(grad_b(C,X,y,w,b)[0])
        b = b - eta*(grad_b(C,X,y,w,b)[1])
        costnew=cost_b(C,X,y,w,b)
        d=(100*np.abs(costold-costnew)/costold)
        cost_list.append(costold)
        k = k + 1
    print(k)
    return(cost_list)


############################################# Running and Ploting 
svm_ba=svm_b(X,y,w,b)
svm_ba=[i[0][0] for i in svm_ba]
fig, ax = plt.subplots()
ax.plot(svm_ba)
ax.set(xlabel='k # of iterations', ylabel='Loss Function',title='Batch Gradient Descent')
fig.savefig("HW2_1.eps",format="eps")
plt.show()












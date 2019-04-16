#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:18:20 2019

@author: shtabari
"""

import numpy as np
import matplotlib.pyplot as plt

#W1=np.loadtxt(open("data/initial_W1.csv", "rb"), delimiter=",")
#W2=np.loadtxt(open("data/initial_W2.csv", "rb"), delimiter=",")


def logf(X):
    sigma=np.zeros(np.shape(X))
    for i in range(0,np.size(sigma,0)):
        for j in range(0,np.size(sigma,1)):
            sigma[i,j]= 1/(1+np.exp(-X[i,j]))
    return sigma

def dlogf(X):
    sigma=logf(X)
    return(sigma*(1-sigma))


def accu(Yh,YR):
    a=np.argmax(Yh,1)+1
    return(100*np.sum(a==YR)/(np.size(YR,axis=0)))
    
def loss(W1,W2,Y,Yh):
    lam=3
    m=5000
    reg=(lam/(2*m))*(np.sum((W1*W1)[:,1:])+np.sum((W2*W2)[:,1:])) 
    er1=-Y*np.log(Yh)
    er2=(1-Y)*np.log(1-Yh)
    err=(1/m)*(np.sum(er1-er2))
    return(err+reg)

def forw(X,W1,W2):
    X=np.insert(X, 0, 1, axis=1)
    
    Z1=np.dot(X,W1.T)
    H=logf(Z1)
    H=np.insert(H, 0, 1, axis=1)
    Z2=np.dot(H,W2.T)
    Yh=logf(Z2)
    return(Yh)
    
def back(X,Y,Yh,W1,W2,lam=3):
    m=5000
    B2=Yh - Y
    Z1=np.dot(np.insert(X, 0, 1, axis=1),W1.T)    
    H=logf(Z1)
    H=np.insert(H, 0, 1, axis=1)
    B1=np.dot(B2,W2[:,1:]) * (dlogf(Z1))
    deW2=np.dot(B2.T,H)
    deW1=np.dot(B1.T,np.insert(X, 0, 1, axis=1))
    W2[::,0] = 0
    W1[::,0] = 0
    dW2 = (deW2/m)+(lam/m)*W2
    dW1 = (deW1/m)+(lam/m)*W1    
    return(dW1,dW2)

############## Reading DATA
W1t=np.loadtxt(open("data/W1.csv", "rb"), delimiter=",")
W2t=np.loadtxt(open("data/W2.csv", "rb"), delimiter=",")
W1=np.loadtxt(open("data/initial_W1.csv", "rb"), delimiter=",")
W2=np.loadtxt(open("data/initial_W2.csv", "rb"), delimiter=",")
X=np.loadtxt(open("data/X.csv", "rb"), delimiter=",")
YR=np.loadtxt(open("data/Y.csv", "rb"), delimiter=",")
Y=np.zeros((len(YR),10))
c=0
for i in YR:
    m=int(i)-1
    Y[c,m]=1
    c+=1

######## Debbging Section
Yht=forw(X,W1t,W2t)
print(accu(Yht,YR))
print(loss(W1t,W2t,Y,Yht))

########
learn=500
cost=[]
accur=[]
lrate=0.2
for i in range(learn):
    Yh=forw(X,W1,W2)
    dW1,dW2 = back(X,Y,Yh,W1,W2)
    W2 = W2 - lrate*dW2 # Update W2
    W1 = W1 - lrate*dW1 # Update W1 
    accur.append(accu(Yh,YR))
    cost.append(loss(W1,W2,Y,Yh))
    print(i)
    
######### Ploting Result
fig, ax = plt.subplots()
ax.plot(cost)
ax.set(xlabel='# of iterations', ylabel='Loss Function')
fig.savefig("COST.png",format="png",dpi=600)
plt.show()


fig, ax = plt.subplots()
ax.plot(accur)
ax.set(xlabel='# of iterations', ylabel='Accuracy')
fig.savefig("accu.png",format="png",dpi=600)
plt.show() 

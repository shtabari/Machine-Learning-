#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:28:53 2019

@author: shtabari
"""

import numpy as np
n = 100
n1 = 60
nL = np.array([50, 30, 80])
n1L = np.array([30, 20, 50])

G = []

def Gini(n,n1,nL,n1L):
    n0L = np.subtract(nL,n1L)
    n0 = np.subtract(n,n1)
    nR = 100 - (nL)
    n1R = n1 - (n1L)
    n0R = nR- n1R
    G= (2*n0*n1)/n - (2*n0L*n1L)/nL - (2*n0R*n1R)/nR
    return G
    
#    
for i in range(3):
    G.append(Gini(n,n1,nL[i],n1L[i]))
print(G)

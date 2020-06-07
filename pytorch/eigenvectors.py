# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:31:13 2020

@author: guru_
"""
import matplotlib.pyplot as plt
import numpy as np
A = np.array([[0.5,0],[0,-1.5]])
[d,v] = np.linalg.eig(A)
v0 = np.array([[0.1],[0.2]])
for i in range(10):
    v0 = A@v0
    print(v0)
    print(np.linalg.norm(v0))
    plt.plot([v0[0],0],[v0[1],0])
    plt.show()
    plt.pause(0.1)
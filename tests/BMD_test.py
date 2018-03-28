import numpy as np
import pandas as pd

import scipy as sp

X = np.loadtxt(open(r'C:\Users\csprock\Documents\Projects\Data Journalism\Article Clustering\BMD\BMD_test.csv', 'rb'), delimiter = ',', skiprows = 1)
Xd = pd.DataFrame(X)

D = np.zeros((6,2))
D[4,1] = 1
D[1,0] = 1
D = pd.DataFrame(D)
#run_clustering(D,X)

W = pd.read_csv('C:/Users/csprock/Documents/Projects/BMD/general/zoo_data.csv')
W = W.values


A = np.zeros((30, 3))
for i in range(30):
    A[i, np.random.randint(0,3)] = 1



B, Y = updateB(A,W)
A = updateA(A, B, W)



run_BMD_bd(A,W,verbose = 1)



A = np.ones((10,3))
B = np.ones((10,2))
C = np.ones((10,4))

W = sp.linalg.block_diag(A,B,C)
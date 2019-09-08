



import bmdcluster

import pandas as pd

# import test data
zoo_data = pd.read_csv('C:/Users/csprock/Documents/Projects/bmdcluster/tests/zoo_data.csv')
class_labels = zoo_data.type.values - 1
W = zoo_data.iloc[:,1:22].values


n, m = W.shape

#### initialize_feature_clusters() ###
#
## testing for correct number of clusters (should = m)
#B_test = initialize_feature_clusters(m, B_ident = True)
#B_test.sum()
#
## test missing keyword argument 
#initialize_feature_clusters(m, B_ident = False)
#
## test feature_clusters > m
#initialize_feature_clusters(m, B_ident = False, feature_clusters = m + 1)
#
#### initialize_data_clusters() (without bootstrapping features) ###
#
## testing for correct number of assignments (should = n)
#A_test = initialize_data_clusters(n, data_clusters = 3)
#A_test.sum()
#
## test data_clusters > n
#initialize_data_clusters(n, data_clusters = n + 1)

### testing bootstrapping functions ###

# testing error handling when number samples to take exceeds total dataset
bootstrap_data(100, 101)

# testing internals of initialize_bootstrapped_clusters()
x_samp, x_rep = bootstrap_data(n, 10)

A_boot_test = initialize_data_clusters(n, data_clusters = 7)
B_boot_test = initialize_feature_clusters(m, B_ident = True)

cost, A_boot, _ = run_BMD(A_boot_test,B_boot_test, W[x_rep, :], verbose = 0)

assign_bootstrapped_clusters(A_boot, x_rep, x_samp)

# testing initialize_bootstrapped_clusters()
c_list = initialize_bootstrapped_clusters(W, data_clusters = 7, block_diag = False, B_ident = True, b = 10)
print(c_list)

# checking missing keyword argument for passing to initialize_feature_clusters()
initialize_bootstrapped_clusters(W, data_clusters = 7, block_diag = False, B_ident = False, b = 10)

# checking missing keyword argument b
initialize_bootstrapped_clusters(W, data_clusters = 7, block_diag = False, B_ident = True)


### initialize_clusters() ###

# error testing of initialize_clusters()
initialize_clusters(W, data_clusters = 101, block_diag = False, use_bootstrap = False)
initialize_clusters(W, data_clusters = 7, block_diag = False, B_ident = False)
initialize_clusters(W, data_clusters = 7, block_diag = False, B_ident = False, use_bootstrap = True, feature_clusters = 15)



######################################
#test_data = pd.read_csv('C:/Users/csprock/Documents/Projects/BMD/general/test_data.csv')
#
#W = test_data.values
#
#A_init = np.zeros((12,3))
#A_init[1,0] = 1
#A_init[5,1] = 1
#A_init[9,2] = 1
#A_init[10,2] = 1
#
#B_init = np.zeros((3,3))
#B_init[0,0] = 1
#B_init[1,1] = 1
#B_init[2,2] = 1

from sklearn.metrics import confusion_matrix


A_init, B_init = initialize_clusters(W, data_clusters = 7, block_diag = False, use_bootstrap = True, feature_clusters = 16, b = 15)

A,B = run_BMD(A_init, B_init, W)
        

C = confusion_matrix(class_labels, np.argmax(A, axis = 1))
C.T



import os, sys
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

if __name__ == '__main__':
    
    mypath = os.path.dirname(os.path.realpath('__file__'))
    sys.path.append(os.path.join(mypath, os.pardir))
    
from bmdcluster import BMD    

#########################################
#############  Zoo Dataset ##############
#########################################

zoo = pd.read_csv('./data/zoo.csv', sep = ',', index_col = 0)

class_labels = zoo.type.values - 1   # get class labels of animal types
zoo = zoo.iloc[:, 0:21]              # remove labels from data

# fit BMD model
BMD_model = BMD(n_clusters = 7, method = 'general', B_ident = True, use_bootstrap = True, b = 10)
BMD_model.fit(zoo.values, verbose = 1)

# show confusion matrix
print(confusion_matrix(class_labels, np.argmax(BMD_model.A, axis = 1)))



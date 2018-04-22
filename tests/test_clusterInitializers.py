import unittest
import numpy as np
import os, sys

if __name__ == '__main__':
    
    mypath = os.path.dirname(os.path.realpath('__file__'))
    sys.path.append(os.path.join(mypath, os.pardir))
        
    from bmdcluster.initializers.clusterInitializers import initializeA, initializeB, MissingKeywordArgument


class TestClusterInitializers(unittest.TestCase):
    
        
    def test_initializeB(self):
        
        m = 3  # set number of features
        
        with self.subTest('Check B_ident'):
            # Check feature cluster matrix B is initialized to identity when B_ident set to True. 
            B = initializeB(m, B_ident = True)
            self.assertTrue(np.array_equal(np.identity(m), B))
            
        
        with self.subTest('Check MissingKeywordArgument'):
            # Check that MissingKeywordArgument raised when B_ident set to False and
            # without additional keyword arguments. 
            with self.assertRaises(MissingKeywordArgument):
                initializeB(m, B_ident = False)
                
        with self.subTest('Check assertions'):
            # Check that when feature_clusters is passed, that AssertionError is raised
            # feature_clusters is not in the interval (1, m]. 
            
            with self.assertRaises(AssertionError):
                initializeB(m, B_ident = False, feature_clusters = m + 1)
        
            with self.assertRaises(AssertionError):
                initializeB(m, B_ident = False, feature_clusters = 1)
                
        
    def test_initializeA(self):
        
        n = 4  # set number of data points
        
        with self.subTest('Check sum of entries'):
            # When init_ratio not set, each point should be assigned exactly one cluster.
            # Check sum of elements of cluster assignment matrix A. 
            A = initializeA(n, n-1)
            self.assertEqual(A.sum(), n)
        
        
        with self.subTest('Check data_cluster size assertion'):
            # Check that assertion error raised when the number of data clusters is greater 
            # than or equal to the size of the dataset. 
            with self.assertRaises(AssertionError):
                initializeA(n = n, data_clusters = n)
                
        with self.subTest('Check init_ratio'):
            # Check that when init_ratio is set, the number of assigned clusters is 
            # the expected number. 
            A = initializeA(n = n, data_clusters = n - 1, init_ratio = 0.5)
            self.assertEqual(A.sum(), n // 2)
            
        with self.subTest('Check init_ratio assertions'):
            
            # Check that assertion error is raised when the init_ratio is outside of 
            # the interval (0,1]. 
            
            with self.assertRaises(AssertionError):
                initializeA(n = n, data_clusters = n - 1, init_ratio = 1.1)
                
            with self.assertRaises(AssertionError):
                initializeA(n = n, data_clusters = n - 1, init_ratio = 0)
                
                
        with self.subTest('Check bootstrap list passing'):
            # Test passing of list of tuples containing the positions of entries 
            # to be set in the returned matrix. 
            
            A_expected = np.array([[1,0],
                                   [0,0],
                                   [0,1],
                                   [0,0]])
            
            
            A = initializeA(n = n, data_clusters = 2, bootstrap = [(0,0),(2,1)])
            self.assertTrue(np.array_equal(A, A_expected))
        
        
        
      
if __name__ == '__main__':
    unittest.main()
    
    
import unittest
import numpy as np

from bmdcluster.initializers.primaryInitializer import initializeClusters



class TestInitializeClusters(unittest.TestCase):

    def test_missing_b_assertion(self):
        '''
        Test that if 'use_bootstrap' is True, then if 'b' is missing, then raise a ValueError
        '''

        with self.assertRaises(ValueError):
            initializeClusters(W=None, n_clusters=None, method='block_diagonal', init_ratio=None, B_ident=False, use_bootstrap=True, b=None)



if __name__ == '__main__':
    unittest.main()

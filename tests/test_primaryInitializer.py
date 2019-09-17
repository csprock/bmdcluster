import unittest
import numpy as np

from bmdcluster.initializers.primary_initializer import initialize_clusters


class Testinitialize_clusters(unittest.TestCase):

    @unittest.skip("This has been moved to the UI modules")
    def test_missing_b_assertion(self):
        '''
        Test that if 'use_bootstrap' is True, then if 'b' is missing, then raise a ValueError
        '''

        with self.assertRaises(ValueError):
            initialize_clusters(W=None, n_clusters=None, method='block_diagonal', init_ratio=None, B_ident=False, use_bootstrap=True, b=None)


if __name__ == '__main__':
    unittest.main()
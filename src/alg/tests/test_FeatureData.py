import unittest
import h5py
import os
import numpy as np
from ddt import ddt, data
from math import prod

from FeatureDataH5 import FeatureDataH5
from datasets_names import FEAT_DSET_NAME

concrete_classes = (FeatureDataH5)


@ddt
class Test_FeatureDataAll(unittest.TestCase):

    # Hypercube data info
    hypercube_test_shape = (3, 4, 5)
    # chunk_test_shape = (3, 2, 5)
    # POROSITY_DSET_NAME = 't'
    # wells_list = [(0, 2), (2, 3)]
    FEATURE_FILENAME = 'test_feature.h5'

    # Porosity data structures
    feature_h5_f = None
    feature_h5_dset = None

    @classmethod
    def setUpClass(cls):
        # Create the feature h5 file
        cls.feature_h5_f = h5py.File(cls.FEATURE_FILENAME, 'w')

        # Create the dataset within the h5
        cls.feature_h5_dset = cls.feature_h5_f.create_dataset(
            FEAT_DSET_NAME,
            cls.hypercube_test_shape,
            dtype=np.float64,
        )

        # Initialize feature data
        for i in range(cls.hypercube_test_shape[0]):
            for j in range(cls.hypercube_test_shape[1]):
                for k in range(cls.hypercube_test_shape[2]):
                    cls.feature_h5_dset[i, j, k] = i * j * k

    @classmethod
    def tearDownClass(cls):
        # Delete any old file
        cls.feature_h5_f.close()
        os.remove(cls.FEATURE_FILENAME)

    @data(concrete_classes)
    def test_feature_init(self, test_cls):
        '''
        Test the creation of a new type.
        '''
        # Load class data
        f_filename = self.__class__.FEATURE_FILENAME

        feature = test_cls(f_filename, None)
        self.assertTrue(True)

    @data(concrete_classes)
    def test_filter_coords(self, test_cls):
        '''
        Test the filtering of coordinates.
        '''
        # Load class data
        f_filename = self.__class__.FEATURE_FILENAME

        # Load feature
        feature = test_cls(f_filename, None)

        # Prepare test case
        coords = np.array([(1, 1, 1), (2, 3, 4), (2, 1, 2), (0, 0, 0)])
        f_vals = [prod(c) for c in coords]

        self.assertTrue((feature.filter_coords(coords) == f_vals).all())


if __name__ == '__main__':
    unittest.main()

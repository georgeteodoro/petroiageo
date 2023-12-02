import unittest
import h5py
import common
import os
import numpy as np
from ddt import ddt, data

from TestDataNumpy import TestDataNumpy

concrete_classes = (TestDataNumpy)


@ddt
class Test_TestDataAll(unittest.TestCase):

    # Porosity data info
    hypercube_test_shape = (3, 4, 5)
    chunk_test_shape = (3, 2, 5)
    POROSITY_DSET_NAME = 't'
    wells_list = [(0, 2), (2, 3)]
    POROSITY_FILENAME = 'test_porosity.h5'

    # Porosity data structures
    porosity_h5_f = None
    porosity_h5_dset = None

    @classmethod
    def setUpClass(cls):
        # Create the porosity h5 file
        cls.porosity_h5_f = h5py.File(cls.POROSITY_FILENAME, 'w')

        # Create the dataset within the h5
        porosity_data_type = np.dtype([
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('real', np.int64),
            ('ring', np.int64),
            ('well_id', np.int64),
        ])
        cls.porosity_h5_dset = cls.porosity_h5_f.create_dataset(
            cls.POROSITY_DSET_NAME,
            cls.hypercube_test_shape,
            dtype=porosity_data_type,
            chunks=cls.chunk_test_shape,
        )

        # Initialize all as empty data
        for i in range(cls.hypercube_test_shape[0]):
            for j in range(cls.hypercube_test_shape[1]):
                for k in range(cls.hypercube_test_shape[2]):
                    cls.porosity_h5_dset[i, j,
                                         k] = (i, j, k, 0,
                                               common.RealValues.empty, -1, -1)

        # Fill wells
        for (well_id, (w_x, w_y)) in enumerate(cls.wells_list):
            for k in range(cls.hypercube_test_shape[2]):
                mock_porosity = w_x * w_y * k
                cls.porosity_h5_dset[w_x, w_y,
                                     k] = (w_x, w_y, k, mock_porosity,
                                           common.RealValues.real, 0, well_id)

    @classmethod
    def tearDownClass(cls):
        # Delete any old file
        cls.porosity_h5_f.close()
        os.remove(cls.POROSITY_FILENAME)

    @data(concrete_classes)
    def test_list_init(self, test_cls):
        '''
        Test the creation of a new type.
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        wells_list = self.__class__.wells_list

        td1 = test_cls(n_features=5,
                       features_only=True,
                       wells_list=wells_list,
                       porosity_data=porosity_dset)
        self.assertTrue(True)
        td2 = test_cls(n_features=5,
                       features_only=False,
                       wells_list=wells_list,
                       porosity_data=porosity_dset)
        self.assertTrue(True)

    @data(concrete_classes)
    def test_prepare_porosity(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        wells_list = self.__class__.wells_list
        n_features = 5

        # Create TestData object
        td1 = test_cls(n_features=n_features,
                       features_only=True,
                       wells_list=wells_list,
                       porosity_data=porosity_dset)

        # Prepare first porosity
        td1.prepare_porosity(0)

        # Check if data was inserted
        self.assertTrue(td1._test_data_dict)


if __name__ == '__main__':
    unittest.main()

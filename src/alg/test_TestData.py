import unittest
import h5py
import common
import os
import numpy as np
from ddt import ddt, data

from TestDataNumpy import TestDataNumpy
from data_filter import WellsSingleRingDataFilter

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
                # Well 1 would not have points throughout the whole depth
                if well_id == 1 and (k == 1 or k == 4):
                    continue

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

        f_sel_filter = WellsSingleRingDataFilter(wells_list)

        td1 = test_cls(n_features=5,
                       features_only=True,
                       wells_list=wells_list,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter)
        self.assertTrue(True)
        td2 = test_cls(n_features=5,
                       features_only=False,
                       wells_list=wells_list,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter)
        self.assertTrue(True)

    @data(concrete_classes)
    def test_prepare_porosity_r0(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        wells_list = self.__class__.wells_list
        n_features = 5
        f_sel_filter = WellsSingleRingDataFilter(list(range(len(wells_list))))

        # Create TestData object
        td1 = test_cls(n_features=n_features,
                       features_only=False,
                       wells_list=wells_list,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter)

        # Prepare first porosity
        td1.prepare_porosity(0)

        # Only one ring exists
        self.assertEqual(len(td1._test_data_dict), 1)

        # Check if all points were added
        # Total of 2 full depths, minus 2 non-added points
        # = 2*5-2 = 8
        self.assertEqual(len(td1._test_data_dict[0]), 8)

    @data(concrete_classes)
    def test_update_feature_r0(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        wells_list = self.__class__.wells_list
        n_features = 5
        f_sel_filter = WellsSingleRingDataFilter(list(range(len(wells_list))))

        # Create TestData object
        td1 = test_cls(n_features=n_features,
                       features_only=False,
                       wells_list=wells_list,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter)

        # Prepare first porosity
        td1.prepare_porosity(0)

        # Add first feature
        td1.update_feature(f1)

        # Update feature

        pass



if __name__ == '__main__':
    unittest.main()
